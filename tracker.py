from ultralytics import YOLO
import torch
from collections import defaultdict
from multiprocessing import Process, Queue, Event

from similarity_search import SimilaritySearch



class Tracker:
    def __init__(self, 
                 yolo_model,
                 embedding_model_name, 
                 hash_size=8, 
                 num_tables=10,
                 device="cuda", 
                 verbose=False):
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Please set device='cpu'.")
        if device not in ["cuda", "cpu"]:
            raise ValueError("Invalid device. Please set device='cuda' or device='cpu'.")
        self.verbose = verbose

        self.yolo = YOLO(yolo_model)
        self.yolo.model.to(device)

        self.similarity_search = SimilaritySearch(embedding_model_name, hash_size, num_tables, device, verbose)
        
        # Create the similarity query process so that it can run in the background
        self.monitor = Event()
        self.monitor.set()
        self.query_queue, self.result_queue = Queue(), Queue()
        self.query_processor = Process(target=self.similarity_search.query_processor, args=(self.query_queue, self.result_queue, self.monitor))
        self.query_processor.start()

        # Create the similarity add process so that it can run in the background
        self.add_queue = Queue()
        self.add_processor = Process(target=self.similarity_search.add_processor, args=(self.add_queue, self.monitor))
        self.add_processor.start()

        # Keep track of object ids currently being tracked
        # dict with object id as key and list of images as value
        # if object id is not detected in the frame, compute hash of the images
        self.currently_tracking = defaultdict(lambda: {
            "images": [],
            "label": None
        })
        # dict with object id as key and tracking id as value
        # this is so you can look up track id from matching object id
        self.tracked_ids = defaultdict(lambda: None)
        # dict with tracking id as key and track history as value
        self.track_history = defaultdict(lambda: {
            "xywh": [],
            "label": None
        })
        # history of yolo tracking objects
        self.yolo_track_history = defaultdict(lambda: {
            "xywh": [],
            "label": None
        })
        # tracked on previous frame
        #self.tracked_on_previous_frame = set()
        # queried objects so that we don't query the same object again
        self.queried_objects = set()
        # track id
        self.track_id = 0

    def track(self, image):
        """
        Tracks object with yolo and store object images of detections
        Once the object is not detected, compute hash of the stacked images
        On new detection, query the hash table for similar objects
        If similar object is found, add to the track id accordingly
        If no similar object is found, create a new track id

        Limitation: if object comes back in frame before being processed by add, it will be treated as new object
        """
        results = self.yolo.track(image, persist=True)
        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        object_ids = results[0].boxes.id.int().cpu().tolist()
        if self.verbose:
            print("Track IDs:", object_ids)

        # add any tracked objects that are not detected in the current frame to the add queue
        for object_id in self.currently_tracking.keys():
            if object_id not in object_ids:
                # wait for object to be queried before adding, in case it has been tracked before (we want to use that track id)
                if object_id not in self.queried_objects:
                    tracked_object = self.currently_tracking[object_id]
                    # if object has been tracked before, use that track id
                    if tracked_object["id"]:
                        track_id = tracked_object["id"]
                    else:
                        track_id = self.track_id
                        self.track_id += 1
                    add = {
                        "images": tracked_object["images"],
                        "id": track_id,
                        "label": tracked_object["label"]
                    }
                    # add object to add queue
                    if len(tracked_object["images"]) > 0:
                        self.add_queue.put(add)
                    # remove object id from currently tracking
                    del(self.currently_tracking[object_id])
        
        # get results of similarity query queue
        while sim_result := self.result_queue.get():
            if self.verbose:
                print("Similarity Result:", sim_result)
            track_id, label, object_id = sim_result
            if object_id in self.queried_objects:
                # update track id to match already tracked object
                self.currently_tracking[object_id]["id"] = track_id
                # remove object id from queried objects so when we add to the add queue, it will have the correct track id
                self.queried_objects.remove(object_id)
            self.tracked_ids[object_id] = track_id
            if track_id is not None:
                # add new track history to existing track history
                # add it to object id to distinguish between detections
                # sort by object id to get latest
                self.track_history[track_id]["xywh"][object_id] = self.yolo_track_history[object_id]
            else:
                # set track history to matching yolo track history
                self.track_history[track_id]["label"] = label
                self.track_history[track_id]["xywh"][object_id] = self.yolo_track_history[object_id]

        """
        Add all track history to yolo track history
        Extract object images
        If object id has already being tracked, add the track box to the track history 
            if object is currently being tracked, add object image to currently tracking and update track id
            else add object to currently being tracked
        If object id has not being tracked, 
            if object is currently being tracked, add images
            else add object to query queue 
        """
        """
        TODO: get label of object
        TODO: figure out format to return tracked objects
        """
        for box, object_id in zip(boxes, object_ids):
            x, y, w, h = box
            track = self.yolo_track_history[object_id]
            track_box = (float(x), float(y), float(w), float(h))
            track.append(track_box)  # x, y center point
            # get extracted object image
            object_image = self.extract_objects(image, box)
            # if object id is already being tracked, add the track box to the track history
            if self.tracked_ids[object_id] is not None:
                track_id = self.tracked_ids[object_id]
                self.track_history[track_id]["xywh"][object_id].append(track_box)
                # add object image to currently tracking
                if object_id in self.currently_tracking:
                    self.currently_tracking[object_id]["images"].append(object_image)
                    self.currently_tracking[object_id]["id"] = track_id
                else:
                    # add object to currently being tracked
                    self.currently_tracking[object_id] = {
                        "images": [object_image],
                        "label": None,
                        "id": track_id
                    }
            else:
                if object_id in self.currently_tracking:
                    self.currently_tracking[object_id]["images"].append(object_image)
                else:
                    self.currently_tracking[object_id] = {
                        "images": [object_image],
                        "label": None,
                        "id": None
                    }
                    # add object id to query queue
                    self.query_queue.put({
                        "image": object_image,
                        "id": object_id
                    })
                    # add object id to queried objects
                    self.queried_objects.add(object_id)


    def extract_objects(self, image, box):
        """Extracts a detected object from an image."""
        x, y, w, h = box
        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
        return image[y1:y2, x1:x2]

    def show(self):
        """Displays the tracked objects."""
        pass

    def stop(self):
        """Stops the similarity query and add processess."""
        self.monitor.clear()
        self.query_processor.terminate()
        self.query_processor.join()
        self.add_processor.terminate()
        self.add_processor.join()
   
    def __del__(self):
        self.stop()