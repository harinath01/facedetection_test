import {
  FaceDetector,
  FilesetResolver,
  Detection,
} from "@mediapipe/tasks-vision";

class FaceDetectionApp {
    video: HTMLVideoElement;
    warningIndicator: HTMLDivElement;
    warningText: HTMLSpanElement;
    videoOverlay: HTMLDivElement;
    webcamCard: HTMLDivElement;
    liveView: HTMLDivElement;
    faceDetector: FaceDetector | null;
    children: HTMLElement[];

    constructor(videoElementId: string) {
        const videoElement = document.getElementById(videoElementId);
        const liveViewElement = document.getElementById("liveView");
        const warningIndicatorElement = document.getElementById("warningIndicator");
        const warningTextElement = document.getElementById("warningText");
        const videoOverlayElement = document.getElementById("videoOverlay");
        const webcamCardElement = document.querySelector(".webcam-card");

        if (!(videoElement instanceof HTMLVideoElement)) {
            throw new Error(`Element with ID ${videoElementId} is not a valid HTMLVideoElement.`);
        }
        if (!(liveViewElement instanceof HTMLDivElement)) {
            throw new Error(`Element with ID liveView is not a valid HTMLDivElement.`);
        }
        if (!(warningIndicatorElement instanceof HTMLDivElement)) {
            throw new Error(`Element with ID warningIndicator is not a valid HTMLDivElement.`);
        }
        if (!(warningTextElement instanceof HTMLSpanElement)) {
            throw new Error(`Element with ID warningText is not a valid HTMLSpanElement.`);
        }
        if (!(videoOverlayElement instanceof HTMLDivElement)) {
            throw new Error(`Element with ID videoOverlay is not a valid HTMLDivElement.`);
        }
        if (!(webcamCardElement instanceof HTMLDivElement)) {
            throw new Error(`Element with class webcam-card is not a valid HTMLDivElement.`);
        }

        this.video = videoElement;
        this.liveView = liveViewElement;
        this.warningIndicator = warningIndicatorElement;
        this.warningText = warningTextElement;
        this.videoOverlay = videoOverlayElement;
        this.webcamCard = webcamCardElement;
        this.faceDetector = null;
        this.children = [];
    }

    async initialize(): Promise<void> {
        try {
            await this.startWebcam();
            await this.initializeMediaPipes();
            this.startFaceDetection();
        } catch (err) {
            console.error("Initialization error:", err);
        }
    }

    async startWebcam(): Promise<void> {
        const stream = await navigator.mediaDevices.getUserMedia({ video: {} });
        this.video.srcObject = stream;
        await this.video.play();
    }

    async initializeMediaPipes(): Promise<void> {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
        );
        this.faceDetector = await FaceDetector.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite`,
                delegate: "GPU"
            },
            minDetectionConfidence: 0.55,
            minSuppressionThreshold: 0.7,     
            runningMode: "IMAGE"
        });
    }

    startFaceDetection(): void {
        setInterval(() => {
            this.detectFace();
        }, 1000);
    }

    createCanvasElement(videoElement: HTMLVideoElement): HTMLCanvasElement {
        const canvas = document.createElement("canvas");
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        const context = canvas.getContext("2d");
        if (context) {
            context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
            return canvas;
        } else {
            throw new Error("2D context for canvas is null or undefined.");
        }
    }

    async detectFace(): Promise<void> {
        if (!this.faceDetector) return;
        
        const canvas = this.createCanvasElement(this.video);
        const result = await this.faceDetector.detect(canvas);
        const detections = result.detections;

        // Display visual detections
        this.displayImageDetections(detections, this.video);

        // Show anomaly label if detected
        if (detections.length === 0) {
            this.showAnomaly('No person detected');
        } else if (detections.length > 1) {
            this.showAnomaly('Multiple persons detected');
        } else {
            this.hideAnomaly();
        }
    }

    showAnomaly(message: string): void {
        this.warningText.textContent = message;
        this.warningIndicator.classList.remove('hidden');
        this.videoOverlay.classList.remove('hidden');
        this.webcamCard.classList.add('warning');
    }

    hideAnomaly(): void {
        this.warningIndicator.classList.add('hidden');
        this.videoOverlay.classList.add('hidden');
        this.webcamCard.classList.remove('warning');
    }

    displayImageDetections(detections: Detection[], resultElement: HTMLVideoElement): void {
        // Remove any highlighting from previous frame
        for (let child of this.children) {
            this.liveView.removeChild(child);
        }
        this.children.splice(0);

        // Get the displayed video dimensions
        const displayedWidth = resultElement.offsetWidth;
        const displayedHeight = resultElement.offsetHeight;
        const naturalWidth = resultElement.videoWidth;
        const naturalHeight = resultElement.videoHeight;

        // Calculate scale factors for object-fit: cover
        // The video maintains aspect ratio and fills the container
        const videoAspect = naturalWidth / naturalHeight;
        const containerAspect = displayedWidth / displayedHeight;
        
        let scaleX: number;
        let scaleY: number;
        let offsetX = 0;
        let offsetY = 0;

        if (videoAspect > containerAspect) {
            // Video is wider - scale by height, crop width
            scaleY = displayedHeight / naturalHeight;
            scaleX = scaleY;
            const scaledWidth = naturalWidth * scaleX;
            offsetX = (displayedWidth - scaledWidth) / 2;
        } else {
            // Video is taller - scale by width, crop height
            scaleX = displayedWidth / naturalWidth;
            scaleY = scaleX;
            const scaledHeight = naturalHeight * scaleY;
            offsetY = (displayedHeight - scaledHeight) / 2;
        }

        for (let detection of detections) {
            if (!detection.boundingBox) continue;

            const score = typeof detection.categories[0].score === 'number' 
                ? detection.categories[0].score 
                : parseFloat(String(detection.categories[0].score));

            // Calculate positions accounting for scaling and offset
            const left = detection.boundingBox.originX * scaleX + offsetX;
            const top = detection.boundingBox.originY * scaleY + offsetY;
            const width = detection.boundingBox.width * scaleX;
            const height = detection.boundingBox.height * scaleY;

            // Confidence text
            const p = document.createElement("p");
            p.setAttribute("class", "info");
            p.innerText =
                "Confidence: " +
                Math.round(score * 100) +
                "% .";
            p.style.left = left + "px";
            p.style.top = (top - 30) + "px";
            p.style.width = (width - 10) + "px";

            // Bounding box highlighter
            const highlighter = document.createElement("div");
            highlighter.setAttribute("class", "highlighter");
            highlighter.style.left = left + "px";
            highlighter.style.top = top + "px";
            highlighter.style.width = width + "px";
            highlighter.style.height = height + "px";

            this.liveView.appendChild(highlighter);
            this.liveView.appendChild(p);
            this.children.push(highlighter);
            this.children.push(p);

            // Keypoints
            if (detection.keypoints) {
                for (let keypoint of detection.keypoints) {
                    const keypointEl = document.createElement("span");
                    keypointEl.className = "key-point";
                    // Keypoints are normalized (0-1), so multiply by natural dimensions then scale
                    const keypointX = keypoint.x * naturalWidth * scaleX + offsetX;
                    const keypointY = keypoint.y * naturalHeight * scaleY + offsetY;
                    keypointEl.style.left = `${keypointX - 3}px`;
                    keypointEl.style.top = `${keypointY - 3}px`;
                    this.liveView.appendChild(keypointEl);
                    this.children.push(keypointEl);
                }
            }
        }
    }
}

// Instantiate and initialize the app
const faceDetectionApp = new FaceDetectionApp("webcam");
faceDetectionApp.initialize();

