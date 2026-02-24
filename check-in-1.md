# Check In 1

## 1. Problem framing + scope

### Clear task definition and success criteria
Given a photograph of a bookshelf, detect and localize individual book spines, extract the visible text, and return a list of identified titles with metadata (title, author, genre, description, cover image).

### A scope that is feasible for the semester
Minimum: Spine detection and localization, text extraction, and metadata retrieval 
Stretch goal: Package the pipeline into a usable web app where anyone can upload a shelf photo and receive a clickable, enriched reading list.

### Modality noted (vision / audio / multimodal)
Vision (with multimodal elements)

## 2. Dataset access + documentation

- Open Shelves (GitHub: capjamesg/open-shelves) - shelf images with spine segmentation annotations, CC BY-NC-SA license, hosted on Roboflow. Downloaded via Roboflow API.
- Library Dataset (GitHub: llabres/library-dataset) - annotated bookshelf images from a public library, includes ISBN/title metadata
- My own photos — Photograph my book shelves

## 3. Data audit / EDA

### Representative samples / visuals / summary stats

### Basic dataset characteristics (classes, balance, etc.)


### Any obvious artifacts, biases, or failure modes


## 4. Evaluation plan

### Metrics appropriate to the task
### Train/validation/test split or CV plan

## 5. Initial baseline / representation

### A first baseline or representation idea (classical features, spectrogram pipeline, basic CNN, etc.)

### Early observations or next steps