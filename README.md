# YoloV8-Object-Tracking-With-Memory
Yolo V8 object tracking, but it recognizes previously tracked objects.  Each detected object's embedding is computed and then stored as a hash.  It uses Locality Sensitive Hashing (LSH) for approximate nearest neighbor storage of hashes.  Similar objects are stored in the same "buckets".
