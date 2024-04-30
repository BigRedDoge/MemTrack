from transformers import AutoFeatureExtractor, AutoModel
import torchvision.transforms as T
import numpy as np
import torch
from typing import List


class SimilaritySearch:
    """
    TODO: Add pickle of lsh table
    """
    def __init__(self, 
                 model_name, 
                 hash_size=8, 
                 num_tables=10,
                 device="cuda", 
                 verbose=False):
        self.verbose = verbose

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
        """Computes hash on a given dataset."""
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

    
    def add(self, batch, label: str):
        """Adds the images to the LSH table."""
        hashes = self.compute_hash(batch)
        for i, h in enumerate(hashes):
            self.lsh.add(i, h, label)

    def query_lsh(self, image):
        # Compute the hashes of the query image and fetch the results.
        batch = dict(image=[image])
        hash = self.compute_hash(batch)["hashes"][0]
        results = self.lsh.query(hash)
        if self.verbose:
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
    
    def query(self, image):
        # Query the LSH table
        counts = self.query_lsh(image)
        # Get the track id with the highest count
        top_result = sorted(counts, key=counts.get, reverse=True)[0]
        track_id, label = top_result.split("_")[0], top_result.split("_")[1]
        return track_id, label


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