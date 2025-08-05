
We provide the code for constructing the CI-VID dataset, including the following key functions:

- **similarity_based_segment** – segments raw videos based on visual similarity.
- **entity_based_segment** – segments videos based on the main entity.
- **individual_annotation** – generates clip-level captions independently.
- **joint_annotation** – generates inter-clip relation captions that capture transitions and narrative coherence.

Our code is built upon the distributed framework [Ray](https://github.com/ray-project/ray). Note the functions are decoupled from Ray’s base classes, thus users may either utilize the Ray-based parallelism or directly instantiate the corresponding classes for single-machine usage.
