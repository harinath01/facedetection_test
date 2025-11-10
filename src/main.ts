import {
  FaceDetector,
  FilesetResolver,
  Detection,
} from "@mediapipe/tasks-vision";

class FaceDetectionApp {
    video: HTMLVideoElement;
    anomalyList: HTMLUListElement;
    liveView: HTMLDivElement;
    faceDetector: FaceDetector | null;
    children: HTMLElement[];

    constructor(videoElementId: string, anomalyListId: string) {
        const videoElement = document.getElementById(videoElementId);
        const anomalyElement = document.getElementById(anomalyListId);
        const liveViewElement = document.getElementById("liveView");

        if (!(videoElement instanceof HTMLVideoElement)) {
            throw new Error(`Element with ID ${videoElementId} is not a valid HTMLVideoElement.`);
        }
        if (!(anomalyElement instanceof HTMLUListElement)) {
            throw new Error(`Element with ID ${anomalyListId} is not a valid HTMLUListElement.`);
        }
        if (!(liveViewElement instanceof HTMLDivElement)) {
            throw new Error(`Element with ID liveView is not a valid HTMLDivElement.`);
        }

        this.video = videoElement;
        this.anomalyList = anomalyElement;
        this.liveView = liveViewElement;
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

        // Keep anomaly list functionality
        if (detections.length === 0) {
            this.addAnomaly('No person detected');
        } else if (detections.length > 1) {
            this.addAnomaly('Multiple persons detected');
        } else {
            this.addAnomaly('Person detected');

            if (this.isFaceTurned(detections[0])) {
                this.addAnomaly('Face turned');
            }
        }
    }

    isFaceTurned(detection: any): boolean {
        const landmarks = detection.keypoints;
        const imageWidth = detection.boundingBox.width;
        const leftEyeX = landmarks[0].x * imageWidth;
        const rightEyeX = landmarks[1].x * imageWidth;
        const jawLeftX = landmarks[4].x * imageWidth;
        const jawRightX = landmarks[5].x * imageWidth;
        const faceWidth = jawRightX - jawLeftX;
        const faceCenterX = (jawLeftX + jawRightX) / 2;

        const threshold = 0.15;
        return (
            Math.abs(leftEyeX - faceCenterX) < faceWidth * threshold ||
            Math.abs(rightEyeX - faceCenterX) < faceWidth * threshold
        );
    }

    addAnomaly(type: string): void {
        const timestamp = new Date().toLocaleTimeString();
        const li = document.createElement('li');
        li.textContent = `${timestamp}: ${type}`;
        this.anomalyList.appendChild(li);
        this.anomalyList.scrollTop = this.anomalyList.scrollHeight;
    }

    displayImageDetections(detections: Detection[], resultElement: HTMLVideoElement): void {
        // Remove any highlighting from previous frame
        for (let child of this.children) {
            this.liveView.removeChild(child);
        }
        this.children.splice(0);

        const ratio = resultElement.height / resultElement.videoHeight;

        for (let detection of detections) {
            if (!detection.boundingBox) continue;

            const score = typeof detection.categories[0].score === 'number' 
                ? detection.categories[0].score 
                : parseFloat(String(detection.categories[0].score));

            // Confidence text
            const p = document.createElement("p");
            p.setAttribute("class", "info");
            p.innerText =
                "Confidence: " +
                Math.round(score * 100) +
                "% .";
            p.style.left = detection.boundingBox.originX * ratio + "px";
            p.style.top = (detection.boundingBox.originY * ratio - 30) + "px";
            p.style.width = (detection.boundingBox.width * ratio - 10) + "px";

            // Bounding box highlighter
            const highlighter = document.createElement("div");
            highlighter.setAttribute("class", "highlighter");
            highlighter.style.left = detection.boundingBox.originX * ratio + "px";
            highlighter.style.top = detection.boundingBox.originY * ratio + "px";
            highlighter.style.width = detection.boundingBox.width * ratio + "px";
            highlighter.style.height = detection.boundingBox.height * ratio + "px";

            this.liveView.appendChild(highlighter);
            this.liveView.appendChild(p);
            this.children.push(highlighter);
            this.children.push(p);

            // Keypoints
            if (detection.keypoints) {
                for (let keypoint of detection.keypoints) {
                    const keypointEl = document.createElement("span");
                    keypointEl.className = "key-point";
                    keypointEl.style.top = `${keypoint.y * resultElement.height - 3}px`;
                    keypointEl.style.left = `${keypoint.x * resultElement.width - 3}px`;
                    this.liveView.appendChild(keypointEl);
                    this.children.push(keypointEl);
                }
            }
        }
    }
}

// Instantiate and initialize the app
const faceDetectionApp = new FaceDetectionApp("webcam", "anomalyList");
faceDetectionApp.initialize();

