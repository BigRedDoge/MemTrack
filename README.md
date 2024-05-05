# Multi-Object Tracking with Memory
Multi-object tracking with YOLOv8, but it memorizes previously tracked objects and recognizes them.  Each detected object's embedding is computed and then stored as a hash.  It uses Locality Sensitive Hashing (LSH) for approximate nearest neighbor search of hashes.  Similar objects are stored in the same "buckets".

![LSH](https://camo.githubusercontent.com/48160bb0db34a86c3a6f3d31c58439eeb681fce0183916d79151d954b69ec67b/68747470733a2f2f64333377756272666b69306c36382e636c6f756466726f6e742e6e65742f356630653765373962333237363931306461343631346633373433326236336137643232366465662f64653265302f696d616765732f6c6f63616c6974792d73656e7369746976652d68617368696e672d31322e6a706567)
