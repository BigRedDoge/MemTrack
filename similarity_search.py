from transformers import AutoFeatureExtractor, AutoModel
import torchvision.transforms as T
import numpy as np
import torch
from typing import List
import time


class SimilaritySearch:
    """
    TODO: Add pickle of lsh table
    TODO: Add parameter for distance to count as collision
    """
    def __init__(self, 
                 model_name, 
                 hash_size=8, 
                 num_tables=10,
                 max_add_images=10,
                 device="cuda",
                 verbose=False):
        self.verbose = verbose
        self.max_add_images = max_add_images

        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.hidden_dim = self.model.config.hidden_size

        self.random_vectors = np.random.randn(hash_size, self.hidden_dim).T

        self.transformation_chain = T.Compose([
            T.Resize(int((256 / 224) * self.extractor.size["height"])),
            T.CenterCrop(self.extractor.size["height"]),
            T.ToTensor(),
            T.Normalize(mean=self.extractor.image_mean, std=self.extractor.image_std),
        ])

        self.lsh = LSH(hash_size, num_tables)

        self.embeddings = np.array([])
        self.embedding_ids = []

    def hash_func(self, embedding):
        """Randomly projects the embeddings and then computes bit-wise hashes."""
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        if len(embedding.shape) < 2:
            embedding = np.expand_dims(embedding, 0)

        # Random projection.
        bools = np.dot(embedding, self.random_vectors) > 0
        return [self._bool2int(bool_vec) for bool_vec in bools]
    
    def _bool2int(self, x):
        y = 0
        for i, j in enumerate(x):
            if j:
                y += 1 << i
        return y
    
    def compute_hash(self, image_batch):
        """Computes hash on a given batch."""
        device = self.model.device
        # Prepare the input images for the model.
        batch = dict(image=image_batch)
        image_batch_transformed = torch.stack(
            [self.transformation_chain(image) for image in image_batch]
        )
        new_batch = {"pixel_values": image_batch_transformed.to(device)}

        # Compute embeddings and pool them i.e., take the representations from the [CLS]
        # token.
        with torch.no_grad():
            embeddings = self.model(**new_batch).last_hidden_state[:, 0].cpu().numpy()

        # Compute hashes for the batch of images.
        hashes = [self.hash_func(embeddings[i]) for i in range(len(embeddings))]
        batch["hashes"] = hashes
        return batch

    def add(self, batch, ids: int, label: str):
        """Adds the images to the LSH table."""
        hashes = self.compute_hash(batch)["hashes"]
        for i, h in enumerate(hashes):
            self.lsh.add(ids, h, label)
        #self.hash_id += 1

    def query_lsh(self, image):
        """
        Queries the LSH table for similar images.
        Returns the counts of similar images.
        """
        hash = self.compute_hash([image])["hashes"][0]
        results = self.lsh.query(hash)
        if self.verbose:
            print("Hash:", hash)
            print("Matches:", len(results))
        
        # Calculate Jaccard index to quantify the similarity.
        # it's the size of the intersection divided by the size of the union of two label sets
        counts = {}
        for r in results:
            if r["id_label"] in counts:
                counts[r["id_label"]] += 1
            else:
                counts[r["id_label"]] = 1
        for k in counts:
            counts[k] = float(counts[k]) / self.hidden_dim
        return counts
    
    def query(self, image, object_id=None):
        """
        Queries the LSH table for similar images.
        Returns the top result.
        """
        # Query the LSH table
        counts = self.query_lsh(image)
        # Get the track id with the highest count
        top_result = sorted(counts, key=counts.get, reverse=True)
        if len(top_result) == 0:
            return None, None, object_id
        else:
            top_id, label = top_result[0].split("_")[0], top_result[0].split("_")[1]
            return top_id, label, object_id
        
    def query_processor(self, query_queue, result_queue, monitor):
        """Processes the queries in the queue."""
        while monitor.is_set():
            if query := query_queue.get():
                result_queue.put(self.query(query["image"], query["id"]))
            time.sleep(0.05)

    def add_processor(self, add_queue, monitor):
        """Processes the adds in the queue."""
        while monitor.is_set():
            if add := add_queue.get():
                if len(add["images"]) <= self.max_add_images:
                    images = add["images"]
                else:
                    indices = np.linspace(0, len(add["images"]) - 1, self.max_add_images, dtype=int)
                    images = [add["images"][i] for i in indices]
                self.add(images, add["id"], add["label"])
            time.sleep(0.05)
    
class Table:
    """
    Table for storing hash values.
    """
    def __init__(self, hash_size: int):
        self.table = {}
        self.hash_size = hash_size

    def add(self, id: int, hashes: List[int], label: int):
        # Create a unique indentifier.
        entry = {"id_label": str(id) + "_" + str(label)}

        # Add the hash values to the current table.
        for h in hashes:
            if h in self.table:
                self.table[h].append(entry)
            else:
                self.table[h] = [entry]

    def query(self, hashes: List[int]):
        results = []

        # Loop over the query hashes and determine if they exist in
        # the current table.
        for h in hashes:
            if h in self.table:
                results.extend(self.table[h])
        return results
    

class LSH:
    """
    Locality Sensitive Hashing (LSH) for approximate nearest neighbor search.
    Tables store the bitwise hash values and the corresponding image ids.
    Similar images are stored in the same bucket.
    Good explanation image: https://camo.githubusercontent.com/48160bb0db34a86c3a6f3d31c58439eeb681fce0183916d79151d954b69ec67b/68747470733a2f2f64333377756272666b69306c36382e636c6f756466726f6e742e6e65742f356630653765373962333237363931306461343631346633373433326236336137643232366465662f64653265302f696d616765732f6c6f63616c6974792d73656e7369746976652d68617368696e672d31322e6a706567
    """
    def __init__(self, hash_size, num_tables):
        self.num_tables = num_tables
        self.tables = []
        for i in range(self.num_tables):
            self.tables.append(Table(hash_size))

    def add(self, id: int, hash: List[int], label: int):
        for table in self.tables:
            table.add(id, hash, label)

    def query(self, hashes: List[int]):
        results = []
        for table in self.tables:
            results.extend(table.query(hashes))
        return results
    
    
"""
Need to figure out which hash ids coorespond to the tracked object
"""
if __name__ == '__main__':
    from PIL import Image
    import glob
    model_name = "google/vit-base-patch16-224-in21k"
    similarity = SimilaritySearch(model_name, hash_size=7, num_tables=1, device="cuda", verbose=True)
    ricks = glob.glob("*.jpg")
    rick_imgs = []
    from multiprocessing import Process, Event, Queue
    import threading
    from concurrent.futures import ProcessPoolExecutor
    
    for i, rick in enumerate(ricks):
        print("Adding Rick:", rick)
        img = Image.open(rick)
        rick_imgs.append(img)
        if i == len(ricks) - 2:
            print(rick)
            break
    #process = multiprocessing.Process(target=similarity.add, args=(rick_imgs, 0, "rick"))
    #process.start()
    #process.join()
    start = time.time()
    similarity.add(rick_imgs, 0, "rick")
    print("Time to add:", (time.time() - start) * 1000, "ms")
    test_rick = Image.open("rick7.jpg")
    print("Testing Slightly Different Rick:", "rick7.jpg")
    start = time.time()
    ids = similarity.query(test_rick)
    print("Similar Rick IDs:", ids)
    print("Time same process:", (time.time() - start) * 1000, "ms")
    monitor = Event()
    monitor.set()
    query_queue, result_queue = Queue(), Queue()
    process = Process(target=similarity.query_processor, args=(query_queue, result_queue, monitor,))
    process.start()
    #executor = ProcessPoolExecutor(max_workers=2)
    start = time.time()
    #ids = executor.submit(similarity.query, test_rick)
    query_queue.put(test_rick)
    while result_queue.empty():
        pass
        #print("Waiting for results")
        #time.sleep(0.05)
    print("Similar Rick IDs:", result_queue.get())
    #print("Similar Rick IDs:", ids.result())
    print("Time multiprocess:", (time.time() - start) * 1000, "ms")  
    monitor.clear()
    process.terminate()
    process.join()
