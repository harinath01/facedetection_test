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
    configContent: HTMLDivElement;
    systemInfo: HTMLDivElement;
    consoleContent: HTMLDivElement;
    consoleOutput: HTMLDivElement;
    clearConsoleBtn: HTMLButtonElement;
    originalConsole: { [key: string]: any };
    faceDetector: FaceDetector | null;
    children: HTMLElement[];

    constructor(videoElementId: string) {
        const videoElement = document.getElementById(videoElementId);
        const liveViewElement = document.getElementById("liveView");
        const warningIndicatorElement = document.getElementById("warningIndicator");
        const warningTextElement = document.getElementById("warningText");
        const videoOverlayElement = document.getElementById("videoOverlay");
        const webcamCardElement = document.querySelector(".webcam-card");
        const configContentElement = document.getElementById("configContent");
        const systemInfoElement = document.getElementById("systemInfo");
        const consoleContentElement = document.getElementById("consoleContent");
        const consoleOutputElement = document.getElementById("consoleOutput");
        const clearConsoleBtnElement = document.getElementById("clearConsole");

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
        if (!(configContentElement instanceof HTMLDivElement)) {
            throw new Error(`Element with ID configContent is not a valid HTMLDivElement.`);
        }
        if (!(systemInfoElement instanceof HTMLDivElement)) {
            throw new Error(`Element with ID systemInfo is not a valid HTMLDivElement.`);
        }
        if (!(consoleContentElement instanceof HTMLDivElement)) {
            throw new Error(`Element with ID consoleContent is not a valid HTMLDivElement.`);
        }
        if (!(consoleOutputElement instanceof HTMLDivElement)) {
            throw new Error(`Element with ID consoleOutput is not a valid HTMLDivElement.`);
        }
        if (!(clearConsoleBtnElement instanceof HTMLButtonElement)) {
            throw new Error(`Element with ID clearConsole is not a valid HTMLButtonElement.`);
        }

        this.video = videoElement;
        this.liveView = liveViewElement;
        this.warningIndicator = warningIndicatorElement;
        this.warningText = warningTextElement;
        this.videoOverlay = videoOverlayElement;
        this.webcamCard = webcamCardElement;
        this.configContent = configContentElement;
        this.systemInfo = systemInfoElement;
        this.consoleContent = consoleContentElement;
        this.consoleOutput = consoleOutputElement;
        this.clearConsoleBtn = clearConsoleBtnElement;
        this.faceDetector = null;
        this.children = [];
        this.originalConsole = {};

        // Setup console clear button
        this.clearConsoleBtn.addEventListener('click', () => {
            this.clearConsole();
        });

        // Intercept console APIs
        this.interceptConsole();
    }

    async initialize(): Promise<void> {
        try {
            await this.startWebcam();
            await this.initializeMediaPipes();
            await this.displaySystemInfo();
            this.startFaceDetection();
        } catch (err) {
            console.error("Initialization error:", err);
        }
    }

    interceptConsole(): void {
        const methods = ['log', 'error', 'warn', 'info', 'debug'];
        
        methods.forEach(method => {
            this.originalConsole[method] = console[method as keyof Console].bind(console);
            
            (console as any)[method] = (...args: any[]) => {
                // Call original console method
                this.originalConsole[method](...args);
                
                // Display in UI
                this.addConsoleEntry(method, args);
            };
        });
    }

    addConsoleEntry(method: string, args: any[]): void {
        const entry = document.createElement('div');
        entry.className = `console-entry ${method}`;
        
        const timestamp = new Date().toLocaleTimeString();
        const timestampSpan = document.createElement('span');
        timestampSpan.className = 'console-timestamp';
        timestampSpan.textContent = timestamp;
        
        const methodSpan = document.createElement('span');
        methodSpan.className = `console-method ${method}`;
        methodSpan.textContent = method.toUpperCase();
        
        const messageSpan = document.createElement('span');
        messageSpan.className = 'console-message';
        
        // Format the message
        const formattedArgs = args.map(arg => {
            if (typeof arg === 'object') {
                try {
                    return JSON.stringify(arg, null, 2);
                } catch (e) {
                    return String(arg);
                }
            }
            return String(arg);
        }).join(' ');
        
        messageSpan.textContent = formattedArgs;
        
        entry.appendChild(timestampSpan);
        entry.appendChild(methodSpan);
        entry.appendChild(messageSpan);
        
        this.consoleOutput.appendChild(entry);
        
        // Auto-scroll to bottom
        this.consoleOutput.scrollTop = this.consoleOutput.scrollHeight;
        
        // Limit to 500 entries to prevent memory issues
        while (this.consoleOutput.children.length > 500) {
            this.consoleOutput.removeChild(this.consoleOutput.firstChild!);
        }
    }


    clearConsole(): void {
        this.consoleOutput.innerHTML = '';
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


    async displaySystemInfo(): Promise<void> {
        const info = await this.gatherSystemInfo();
        const table = document.createElement('table');
        
        for (const [key, value] of Object.entries(info)) {
            const row = document.createElement('tr');
            const keyCell = document.createElement('td');
            const valueCell = document.createElement('td');
            
            keyCell.textContent = key;
            
            if (typeof value === 'object' && value !== null) {
                if (value.status) {
                    valueCell.innerHTML = `<span class="status-${value.status}">${value.text}</span>`;
                } else {
                    valueCell.textContent = JSON.stringify(value, null, 2);
                }
            } else {
                valueCell.textContent = String(value);
            }
            
            row.appendChild(keyCell);
            row.appendChild(valueCell);
            table.appendChild(row);
        }
        
        this.systemInfo.appendChild(table);
    }

    async gatherSystemInfo(): Promise<Record<string, any>> {
        const info: Record<string, any> = {};

        // Browser Information
        const ua = navigator.userAgent;
        const browserMatch = ua.match(/(Chrome|Firefox|Safari|Edge|Opera)\/(\d+)/);
        info['Browser'] = browserMatch ? `${browserMatch[1]} ${browserMatch[2]}` : ua.split(' ')[0];
        info['User Agent'] = ua;
        info['Platform'] = navigator.platform;
        info['Language'] = navigator.language;
        info['Cookie Enabled'] = navigator.cookieEnabled ? 'Yes' : 'No';

        // Hardware Information
        info['CPU Cores'] = navigator.hardwareConcurrency || null;
        info['Device Memory'] = (navigator as any).deviceMemory ? `${(navigator as any).deviceMemory} GB` : null;

        // Screen Information
        info['Screen Resolution'] = `${screen.width}x${screen.height}`;
        info['Screen Color Depth'] = `${screen.colorDepth} bits`;
        info['Window Size'] = `${window.innerWidth}x${window.innerHeight}`;
        info['Device Pixel Ratio'] = window.devicePixelRatio || 1;

        // WebGL/GPU Information
        const webglInfo = this.getWebGLInfo();
        info['WebGL Support'] = webglInfo.supported ? { status: 'ok', text: 'Yes' } : { status: 'error', text: 'No' };
        if (webglInfo.supported) {
            info['WebGL Renderer'] = webglInfo.renderer || 'Unknown';
            info['WebGL Vendor'] = webglInfo.vendor || 'Unknown';
            info['WebGL Version'] = webglInfo.version || 'Unknown';
            info['WebGL Shading Language'] = webglInfo.shadingLanguage || 'Unknown';
        }

        // MediaPipe Information
        info['GPU Delegate'] = this.faceDetector ? { status: 'ok', text: 'Enabled' } : { status: 'warning', text: 'Not initialized' };

        // Media Devices
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(d => d.kind === 'videoinput');
            info['Video Input Devices'] = videoDevices.length > 0 ? videoDevices.length : { status: 'warning', text: 'None detected' };
            if (videoDevices.length > 0) {
                info['Active Video Device'] = videoDevices[0].label || 'Default';
            }
        } catch (err) {
            info['Video Input Devices'] = { status: 'error', text: 'Access denied' };
        }


        // Video Element Info
        if (this.video.videoWidth > 0) {
            info['Video Resolution'] = `${this.video.videoWidth}x${this.video.videoHeight}`;
            info['Video Aspect Ratio'] = (this.video.videoWidth / this.video.videoHeight).toFixed(2);
        } else {
            info['Video Resolution'] = { status: 'warning', text: 'Not available yet' };
        }

        // Performance
        if ('memory' in performance) {
            const memory = (performance as any).memory;
            info['JS Heap Used'] = `${(memory.usedJSHeapSize / 1048576).toFixed(2)} MB`;
            info['JS Heap Total'] = `${(memory.totalJSHeapSize / 1048576).toFixed(2)} MB`;
            info['JS Heap Limit'] = `${(memory.jsHeapSizeLimit / 1048576).toFixed(2)} MB`;
        }

        // Connection Info (if available)
        if ('connection' in navigator) {
            const conn = (navigator as any).connection;
            if (conn) {
                info['Connection Type'] = conn.effectiveType || 'Unknown';
                info['Connection Downlink'] = conn.downlink ? `${conn.downlink} Mbps` : 'Unknown';
            }
        }

        return info;
    }

    getWebGLInfo(): { supported: boolean; renderer?: string; vendor?: string; version?: string; shadingLanguage?: string } {
        try {
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl') as WebGLRenderingContext | null;
            
            if (!gl) {
                return { supported: false };
            }

            const webglContext = gl as WebGLRenderingContext;
            const debugInfo = webglContext.getExtension('WEBGL_debug_renderer_info');
            if (debugInfo) {
                return {
                    supported: true,
                    renderer: webglContext.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) as string,
                    vendor: webglContext.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL) as string,
                    version: webglContext.getParameter(webglContext.VERSION) as string,
                    shadingLanguage: webglContext.getParameter(webglContext.SHADING_LANGUAGE_VERSION) as string
                };
            }

            return {
                supported: true,
                version: webglContext.getParameter(webglContext.VERSION) as string,
                shadingLanguage: webglContext.getParameter(webglContext.SHADING_LANGUAGE_VERSION) as string
            };
        } catch (err) {
            return { supported: false };
        }
    }
}

// Instantiate and initialize the app
const faceDetectionApp = new FaceDetectionApp("webcam");
faceDetectionApp.initialize();

