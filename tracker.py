from ultralytics import YOLO
import torch
from collections import defaultdict
import multiprocessing as mp
from threading import Thread, Event
from queue import Queue
import cv2
import numpy as np
from PIL import Image
import time

from similarity_search import SimilaritySearch



class Tracker:
    def __init__(self, 
                 yolo_model,
                 embedding_model_name, 
                 hash_size=6, 
                 num_tables=10,
                 max_add_images=10,
                 multiprocessing=False, 
                 device="cuda", 
                 verbose=False):
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Please set device='cpu'.")
        if device not in ["cuda", "cpu"]:
            raise ValueError("Invalid device. Please set device='cuda' or device='cpu'.")
        if multiprocessing:
            raise NotImplementedError("Multiprocessing is not supported yet.")
        
        self.verbose = verbose
        self.multiprocessing = multiprocessing
        self.max_add_images = max_add_images

        self.yolo = YOLO(yolo_model)
        self.yolo.model.to(device)

        self.similarity_search = SimilaritySearch(embedding_model_name, hash_size, num_tables, max_add_images, device, verbose)
        
        # Multiprocessing
        if self.multiprocessing:
            self.monitor = mp.Event()
            self.monitor.set()
            # Create the similarity query process so that it can run in the background
            self.query_queue, self.result_queue = mp.Queue(), mp.Queue()
            self.query_processor = mp.Process(target=self.similarity_search.query_processor, args=(self.query_queue, self.result_queue, self.monitor))
            self.query_processor.start()

            # Create the similarity add process so that it can run in the background\
            self.add_queue = mp.Queue()
            self.add_processor = mp.Process(target=self.similarity_search.add_processor, args=(self.add_queue, self.monitor))
            self.add_processor.start()
        else:
            # Queues for adding and querying objects
            self.add_queue, self.query_queue, self.result_queue = Queue(), Queue(), Queue()

        # Keep track of object ids currently being tracked
        # dict with object id as key and list of images as value
        # if object id is not detected in the frame, compute hash of the images
        self.currently_tracking = defaultdict(lambda: {
            "images": [],
            "label": None,
            "id": None
        })
        # dict with object id as key and tracking id as value
        # this is so you can look up track id from matching object id
        self.tracked_ids = defaultdict(lambda: None)
        # dict with tracking id as key and track history as value
        self.track_history = defaultdict(lambda: {
            "xywh": defaultdict(lambda: []),
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
        # labels of tracked objects, key is track id
        self.track_labels = {}
        self.total_time = 0
        self.count = 0
        self.max_time = 0

    def track(self, image, conf=0.3, iou=0.5):
        """
        TODO: continue to check if new object has previously been tracked, if so, update track id
        TODO: fix multiprocessing, it literally blue screens my computer
        TODO: remove id from currently_tracking, it's not used
        It's written "interestingly" with queues because I was trying to get multiprocessing to work
        """
        """
        Tracks object with yolo and store object images of detections
        Once the object is not detected, compute hash of the stacked images
        On new detection, query the hash table for similar objects
        If similar object is found, add to the track id accordingly
        If no similar object is found, create a new track id

        Limitation: if object comes back in frame before being processed by add, it will be treated as new object
        """
        start = time.time()
        if not self.multiprocessing:
            while not self.query_queue.empty():
                query = self.query_queue.get()
                result = self.similarity_search.query(query["image"], query["id"])
                self.result_queue.put(result)

            while not self.add_queue.empty():
                add = self.add_queue.get()
                if len(add["images"]) <= self.max_add_images:
                    images = add["images"]
                else:
                    indices = np.linspace(0, len(add["images"]) - 1, self.max_add_images, dtype=int)
                    images = [add["images"][i] for i in indices]
                self.similarity_search.add(images, add["id"], add["label"])
            
        # Get the results of the YOLO model
        results = self.yolo.track(image, conf=conf, iou=iou, persist=True)
        #print("Results:", results[0].boxes)
        
        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        cls = results[0].boxes.cls.cpu().tolist()
        class_names = [self.yolo.names[i] for i in cls]

        if results[0].boxes.id is None:
            print("No objects detected")
            return Result(None, (results[0].boxes.xywh.cpu(), class_names), image)
        
        object_ids = results[0].boxes.id.int().cpu().tolist()
        object_labels = {object_id: class_names[i] for i, object_id in enumerate(object_ids)}
        #for object_id, label in track_labels.items():
        #    self.track_labels[object_id] = label

        if self.verbose:
            print("Track IDs:", object_ids)

        # add any tracked objects that are not detected in the current frame to the add queue
        tracking_keys = list(self.currently_tracking.keys())
        for object_id in tracking_keys:
            if object_id not in object_ids:
                # wait for object to be queried before adding, in case it has been tracked before (we want to use that track id)
                if object_id not in self.queried_objects:
                    tracked_object = self.currently_tracking[object_id]
                    track_id = self.tracked_ids[object_id]
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
                    # add track history
                    self.track_history[track_id]["label"] = tracked_object["label"]
                    self.track_history[track_id]["xywh"][object_id] = self.yolo_track_history[object_id]["xywh"]

        # get results of similarity query queue
        while not self.result_queue.empty():
            sim_result = self.result_queue.get()
            if self.verbose:
                print("Similarity Result:", sim_result)
            track_id, label, object_id = sim_result
            # if object is currently being tracked, update track id and add history
            if track_id is not None:
                #if object_id in self.queried_objects:
                # update track id to match already tracked object
                #self.currently_tracking[object_id]["id"] = track_id
                # remove object id from queried objects so when we add to the add queue, it will have the correct track id
                #self.queried_objects.remove(object_id)
                self.tracked_ids[object_id] = track_id
                # add new track history to existing track history
                # add it to object id to distinguish between detections
                # sort by object id to get latest
                # set track history to matching yolo track history
                self.track_history[track_id]["label"] = label
                self.track_history[track_id]["xywh"][object_id] = self.yolo_track_history[object_id]["xywh"]
            else:
                # if no similar object is found, create a new track id
                self.currently_tracking[object_id]["id"] = object_id
                self.tracked_ids[object_id] = object_id
            # remove object id from queried objects so when we add to the add queue, it will have the correct track id
            self.queried_objects.remove(object_id)

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
        for box, object_id in zip(boxes, object_ids):
            x, y, w, h = box
            track = self.yolo_track_history[object_id]
            # if no label, add label to yolo track history
            if not self.yolo_track_history[object_id]["label"]:
                self.yolo_track_history[object_id]["label"] = object_labels[object_id]
            track_box = (float(x), float(y), float(w), float(h))
            # add track box to yolo track history
            track["xywh"].append(track_box)  # x, y center point
            # get extracted object image
            object_image = self.extract_objects(image, box)
            # if object id is already being tracked, add the track box to the track history\
            if self.tracked_ids[object_id] is not None:
                track_id = self.tracked_ids[object_id]
                if object_id not in self.track_history[track_id]["xywh"]:
                    self.track_history[track_id]["xywh"][object_id] = []
                self.track_history[track_id]["xywh"][object_id].append(track_box)
                # add object image to currently tracking
                if object_id in self.currently_tracking:
                    self.currently_tracking[object_id]["images"].append(object_image)
                    self.currently_tracking[object_id]["id"] = track_id
                else:
                    # add object to currently being tracked
                    self.currently_tracking[object_id] = {
                        "images": [object_image],
                        "label": object_labels[object_id],
                        "id": track_id
                    }
            else:
                self.tracked_ids[object_id] = object_id
                if object_id in self.currently_tracking:
                    self.currently_tracking[object_id]["images"].append(object_image)
                else:
                    self.currently_tracking[object_id] = {
                        "images": [object_image],
                        "label": object_labels[object_id],
                        "id": None
                    }
                    # add object to query queue
                    self.query_queue.put({
                        "image": object_image,
                        "id": object_id
                    })
                    # add object id to queried objects
                    self.queried_objects.add(object_id)
        # track ids for the current frame
        track_ids = {self.tracked_ids[key]: key for key in object_ids}
        # history of current tracked objects in frame
        history = {key: self.track_history[key] for key, value in track_ids.items()}
        track_labels = {key: value["label"] for key, value in history.items()}
        end = time.time()
        self.total_time += end - start
        self.count += 1
        if end - start > self.max_time:
            self.max_time = end - start
        print("Average time:", self.total_time / self.count, "Max time:", self.max_time)
        return Result((history, track_ids, track_labels), (boxes, class_names), image)

    def extract_objects(self, image, box):
        """Extracts a detected object from an image."""
        x, y, w, h = box
        x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
        cropped = Image.fromarray(image[y1:y2, x1:x2])
        return cropped

    def stop(self):
        """Stops the similarity query and add processess."""
        if self.multiprocessing:
            self.monitor.clear()
            self.query_processor.join()
            self.add_processor.join()
            self.query_processor.terminate()
            self.add_processor.terminate()
   
    def __del__(self):
        self.stop()


class Result:
    """Result of tracking objects in a frame."""
    def __init__(self, track, boxes, image):
        if track is not None:
            # history of tracked objects, key is track id
            # track ids for the current frame, key is track id, value is object id
            # labels of tracked objects, key is track id
            self.track_history, self.track_ids, self.track_labels = track
        else:
            self.track_history, self.track_ids, self.track_labels = None, None, None
        # boxes of detected objects
        self.boxes = boxes[0]
        # class names of detected objects
        self.class_names = boxes[1]
        # image with tracked objects
        self.image = image

    def plot(self):
        """Displays the tracked objects."""
        if self.image is None:
            return None
        frame = self.image.copy()
        if self.track_history:
            for track_id, track in self.track_history.items():
                # get the latest track box
                xywh = track["xywh"][self.track_ids[track_id]]
                #for box in xywh:'
                if xywh:
                    x, y, w, h = xywh[-1]
                    x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
                    # draw the bounding boxes
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Draw the tracking lines
                    xy_points = [(int(x), int(y)) for x, y, _, _ in xywh]
                    points = np.hstack(xy_points).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

                    # draw label and track id
                    frame = cv2.putText(frame, track["label"], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    frame = cv2.putText(frame, str(track_id), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw bounding boxes and write class names
        for box, class_name in zip(self.boxes, self.class_names):
            x, y, w, h = box
            x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            frame = cv2.putText(frame, class_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Tracked Objects", frame)

    def __str__(self):
        return str(self.history)
    