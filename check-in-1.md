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
Shown in eda.ipynb

### Basic dataset characteristics (classes, balance, etc.)
There are may be some genres underrepresented in the dataset such as children's books, but if the model can generalize well, it should still be able to handle these cases.

### Any obvious artifacts, biases, or failure modes
Books with very thin spines or books partially obstructed may be hard to read the title/ author from. 

## 4. Evaluation plan

Plan to do a 70/15/15 train/val/test split

Detection: mAP@0.5 and mAP@0.5:0.95 for spine localization

Title match rate — what % of detected spines resolve to the correct book in the Google Books API after OCR

I plan to keep my personal photos entirely in the test set to evaluate generalization to unseen, real-world conditions. 

## 5. Initial baseline / representation

### A first baseline or representation idea (classical features, spectrogram pipeline, basic CNN, etc.)

I plan to use a basic CNN as my first baseline. Then train an object detection system to extact books from any angle and identify with a multimodal transformer.

### Early observations or next steps

Next steps: object detection and segmentation to identify the locations of all the books on the images. 