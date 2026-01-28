document.addEventListener('DOMContentLoaded', () => {
    console.log('Script initialized');

    // --- 1. Init Splitting.js (Text Animation) ---
    if (typeof Splitting !== 'undefined') {
        Splitting();
    }

    // --- 2. GSAP Registration ---
    if (typeof gsap !== 'undefined') {
        if (typeof ScrollTrigger !== 'undefined') gsap.registerPlugin(ScrollTrigger);
        if (typeof ScrollToPlugin !== 'undefined') gsap.registerPlugin(ScrollToPlugin);
        if (typeof Flip !== 'undefined') gsap.registerPlugin(Flip);
    }

    // --- 3. Lenis Smooth Scroll Setup ---
    try {
        if (typeof Lenis !== 'undefined') {
            const lenis = new Lenis({
                lerp: 0.1,
                smooth: true
            });
            lenis.on('scroll', ScrollTrigger.update);
            gsap.ticker.add((time) => {
                lenis.raf(time * 1000);
            });
            gsap.ticker.lagSmoothing(0);
        }
    } catch (e) { console.log("Lenis skipped"); }

    // ===============================================
    // 4. PRELOADER: KINETIC IMPACT (RELOAD AWARE)
    // ===============================================

    // Elements
    const wordLayer = document.getElementById('word-layer');
    const wordContainer = document.getElementById('word-container');
    const currentWord = document.getElementById('current-word');
    const slidePath = document.getElementById('slide-path');

    const eyeLayer = document.getElementById('eye-layer');
    const shutterLid = document.getElementById('shutter-lid');
    const impactGroup = document.getElementById('impact-group');
    const flash = document.getElementById('flash');
    const scanlines = document.querySelector('.scanlines');

    const valX = document.getElementById('valX');
    const valY = document.getElementById('valY');

    // --- LOGIC START ---
    if (wordLayer && eyeLayer) {

        // 1. Detect if this is a Page Reload
        // This checks the browser's performance API to see how we got here
        const navEntries = performance.getEntriesByType("navigation");
        let isReload = false;
        if (navEntries.length > 0 && navEntries[0].type === 'reload') {
            isReload = true;
        }

        // 2. Check Session Storage
        const hasSeenIntro = sessionStorage.getItem('oro_intro_seen');
        const forcePreloader = sessionStorage.getItem('forcePreloader');

        // 3. DECISION MATRIX:
        // IF (We have seen intro) AND (It is NOT a reload) AND (Not forced) -> HIDE
        if (hasSeenIntro && !isReload && forcePreloader !== 'true') {

            // Skip Animation Instantly
            if (wordLayer) wordLayer.classList.add('preloader-hidden');
            if (eyeLayer) eyeLayer.classList.add('preloader-hidden');
            if (scanlines) scanlines.classList.add('preloader-hidden');
            document.body.classList.remove('preloader-active');

            // Cleanup force flag
            sessionStorage.removeItem('forcePreloader');

        } else {
            // PLAY ANIMATION (Because it's either First Visit OR a Reload)

            sessionStorage.removeItem('forcePreloader');
            document.body.classList.add('preloader-active');

            // --- ANIMATION VARIABLES ---
            const words = ['Bonjour', 'Ciao', 'Olà', 'やあ', 'Hallå', 'Hallo', 'नमस्ते', 'హలో', 'வணக்கம்', 'Hello']; let wIndex = 0;
            let width = window.innerWidth;
            let height = window.innerHeight;

            const pathClosed = "M -1000,-1000 L 3000,-1000 L 3000,3000 L -1000,3000 Z";
            const pathOpen = "M -1000,-4000 L 3000,-4000 L 3000,-1000 L -1000,-1000 Z";

            if (shutterLid) shutterLid.setAttribute('d', pathClosed);

            if (valX && valY) {
                setInterval(() => {
                    valX.innerText = Math.floor(Math.random() * 999);
                    valY.innerText = Math.floor(Math.random() * 999);
                }, 30);
            }

            if (wordContainer) wordContainer.classList.add('visible');

            // Word Cycle
            function nextWord() {
                if (wIndex < words.length - 1) {
                    setTimeout(() => {
                        wIndex++;
                        if (currentWord) currentWord.innerText = words[wIndex];
                        nextWord();
                    }, wIndex === 0 ? 1000 : 200);
                } else {
                    setTimeout(() => {
                        if (wordContainer) wordContainer.classList.remove('visible');
                        animateSlideUp();
                    }, 400);
                }
            }
            if (currentWord) nextWord();

            // Slide Up
            function animateSlideUp() {
                let start = null;
                const duration = 600;
                const initialPath = `M0 0 L${width} 0 L${width} ${height} Q${width / 2} ${height + 300} 0 ${height} L0 0`;
                if (slidePath) slidePath.setAttribute('d', initialPath);

                function step(timestamp) {
                    if (!start) start = timestamp;
                    const progress = timestamp - start;
                    const pct = Math.min(progress / duration, 1);
                    const ease = 1 - Math.pow(1 - pct, 4);

                    const moveY = -window.innerHeight * ease;
                    if (wordLayer) wordLayer.style.transform = `translateY(${moveY}px)`;

                    const curveH = 300 * (1 - ease);
                    const currPath = `M0 0 L${width} 0 L${width} ${height} Q${width / 2} ${height + curveH} 0 ${height} L0 0`;
                    if (slidePath) slidePath.setAttribute('d', currPath);

                    if (progress < duration) {
                        requestAnimationFrame(step);
                    } else {
                        triggerEyeOpening();
                        if (wordLayer) wordLayer.style.display = 'none';
                    }
                }
                requestAnimationFrame(step);
            }

            // Eye Impact
            function triggerEyeOpening() {
                setTimeout(() => {
                    if (shutterLid) shutterLid.setAttribute('d', pathOpen);

                    setTimeout(() => {
                        if (impactGroup) impactGroup.classList.add('lock-on');
                        if (eyeLayer) eyeLayer.classList.add('shaking');

                        setTimeout(() => {
                            if (eyeLayer) eyeLayer.classList.remove('shaking');
                            if (eyeLayer) eyeLayer.classList.add('zooming');

                            setTimeout(() => {
                                if (flash) flash.classList.add('active');

                                setTimeout(() => {
                                    if (eyeLayer) eyeLayer.style.display = 'none';
                                    if (scanlines) scanlines.style.display = 'none';
                                    document.body.classList.remove('preloader-active');

                                    // *** Mark as Seen ***
                                    sessionStorage.setItem('oro_intro_seen', 'true');

                                    if (typeof gsap !== 'undefined') {
                                        gsap.from('.hero-content h1', { y: 50, opacity: 0, duration: 1 });
                                    }
                                }, 500);

                            }, 600);
                        }, 400);
                    }, 1000);
                }, 50);
            }
        } // End Else

        // Resize Listener
        window.addEventListener('resize', () => {
            width = window.innerWidth;
            height = window.innerHeight;
        });
    }

    // ===============================================
    // 5. NAVBAR & MENU
    // ===============================================
    const navbar = document.querySelector('.floating-nav');
    let lastScrollY = window.scrollY;

    if (navbar) {
        window.addEventListener('scroll', () => {
            const currentScrollY = window.scrollY;
            if (currentScrollY < 50) {
                navbar.classList.remove('nav-hidden');
                navbar.classList.add('nav-visible');
            } else if (currentScrollY > lastScrollY) {
                navbar.classList.remove('nav-visible');
                navbar.classList.add('nav-hidden');
            } else {
                navbar.classList.remove('nav-hidden');
                navbar.classList.add('nav-visible');
            }
            lastScrollY = currentScrollY;
        });
    }

    const hamburger = document.getElementById('hamburger-menu');
    const navLinks = document.getElementById('nav-links');

    if (hamburger && navLinks) {
        hamburger.addEventListener('click', function () {
            navLinks.classList.toggle('active');
            hamburger.classList.toggle('active');
        });

        document.addEventListener('click', function (event) {
            if (!event.target.closest('.floating-nav')) {
                navLinks.classList.remove('active');
                hamburger.classList.remove('active');
            }
        });

        const navItems = navLinks.querySelectorAll('a');
        navItems.forEach(item => {
            item.addEventListener('click', function () {
                navLinks.classList.remove('active');
                hamburger.classList.remove('active');
            });
        });
    }

    // ===============================================
    // 6. HOME PAGE: HERO ANIMATION
    // ===============================================
    if (document.querySelector('.hero-img') && typeof gsap !== 'undefined') {
        gsap.to('.hero-img', {
            scale: 1,
            scrollTrigger: {
                trigger: '.hero',
                start: 'top top',
                end: 'bottom top',
                scrub: true
            }
        });
    }

    // ===============================================
    // 7. RESPONSIVE HORIZONTAL SCROLL (SIMPLIFIED & FINAL)
    // ===============================================

    if (typeof gsap !== 'undefined' && typeof ScrollTrigger !== 'undefined') {

        ScrollTrigger.matchMedia({

            // --- TABLET & DESKTOP ANIMATION ---
            "(min-width: 769px)": function () {

                const section = document.querySelector(".desktop-hscroll-section");
                const track = document.getElementById('hscroll-track');

                if (!section || !track) return;

                // Set the height of the main trigger section to create scrollable space
                let scrollDistance = track.offsetWidth - window.innerWidth;
                section.style.height = (scrollDistance + window.innerHeight) + 'px';

                // Create the main horizontal scroll animation
                gsap.to(track, {
                    x: -scrollDistance,
                    ease: "none",
                    scrollTrigger: {
                        trigger: section,
                        start: "top top",
                        end: "bottom bottom",
                        scrub: 1,
                        // This onUpdate callback now only handles the vertical parallax
                        onUpdate: self => {
                            const currentPixel = self.progress * scrollDistance;

                            // Vertical parallax for text rails
                            const txtRails = track.querySelectorAll('.txt-rail');
                            txtRails.forEach(rail => {
                                rail.style.transform = `translateY(-${currentPixel * 0.7}px)`;
                            });
                        }
                    }
                });

                // Ensure everything recalculates on window resize
                ScrollTrigger.addEventListener("refresh", () => {
                    scrollDistance = track.offsetWidth - window.innerWidth;
                    section.style.height = (scrollDistance + window.innerHeight) + 'px';
                });
            },

            // --- PHONE ANIMATION ---
            "(max-width: 768px)": function () {
                const impactSection = document.querySelector('.horizontal-impact');
                const impactTrack = document.querySelector('.impact-track');

                if (impactSection && impactTrack) {
                    const tween = gsap.to(impactTrack, {
                        x: () => -(impactTrack.scrollWidth - window.innerWidth),
                        ease: "none",
                    });
                    ScrollTrigger.create({
                        trigger: ".horizontal-impact",
                        start: "top top",
                        end: "bottom bottom",
                        pin: ".impact-wrapper",
                        animation: tween,
                        scrub: 1,
                        invalidateOnRefresh: true,
                    });
                }
            }

        }); // End of matchMedia
    } // End of GSAP check

    // ===============================================
    // 8. RESPONSIVE ANIMATIONS (Desktop Grid vs Mobile Stack)
    // ===============================================
    if (typeof ScrollTrigger !== 'undefined') {

        ScrollTrigger.matchMedia({

            // --- DESKTOP ONLY (> 768px) ---
            "(min-width: 769px)": function () {
                const gridMenuSection = document.querySelector('.grid-menu-section');
                if (!gridMenuSection) return;

                class Row {
                    constructor(DOM_el) {
                        this.DOM = {
                            el: DOM_el,
                            titleWrap: DOM_el.querySelector('.gm-cell__title'),
                            title: DOM_el.querySelector('.gm-cell__title .oh__inner'),
                            images: [...DOM_el.querySelectorAll('.gm-cell__img')]
                        };
                    }
                }

                const rows = [...document.querySelectorAll('.gm-row')];
                if (rows.length > 0) {
                    let rowsArr = rows.map(row => new Row(row));

                    rowsArr.forEach(row => {
                        // Hover Enter
                        row.DOM.el.addEventListener('mouseenter', () => {
                            gsap.killTweensOf([row.DOM.images, row.DOM.title]);

                            const tl = gsap.timeline();
                            tl.fromTo(row.DOM.images,
                                { scale: 0.8, x: 50, opacity: 0 },
                                { duration: 0.4, ease: 'power3.out', scale: 1, x: 0, opacity: 1, stagger: -0.05, overwrite: true }
                            )
                                .to(row.DOM.title, { duration: 0.2, y: '-100%', ease: 'power2.in', onComplete: () => row.DOM.titleWrap.classList.add('gm-cell__title--switch') }, 0)
                                .fromTo(row.DOM.title, { y: '100%', rotation: 10 }, { duration: 0.4, y: '0%', rotation: 0, ease: 'back.out(1.7)' }, 0.2);
                        });

                        // Hover Leave
                        row.DOM.el.addEventListener('mouseleave', () => {
                            gsap.killTweensOf([row.DOM.images, row.DOM.title]);
                            const tl = gsap.timeline();
                            tl.to(row.DOM.images, { duration: 0.3, ease: 'power2.in', opacity: 0, scale: 0.8, x: 20, overwrite: true })
                                .to(row.DOM.title, { duration: 0.2, y: '-100%', ease: 'power2.in', onComplete: () => row.DOM.titleWrap.classList.remove('gm-cell__title--switch') }, 0)
                                .fromTo(row.DOM.title, { y: '100%', rotation: 10 }, { duration: 0.4, y: '0%', rotation: 0, ease: 'power2.out' }, 0.2);
                        });
                    });
                }
            },

            // --- MOBILE ONLY (<= 768px) ---
            "(max-width: 768px)": function () {
                const stackSection = document.querySelector('.stacking-cards-section');
                if (!stackSection) return;

                const cards = gsap.utils.toArray(".stack-card");

                cards.forEach((card, i) => {
                    if (i === cards.length - 1) return; // Don't animate the very last card

                    const nextCard = cards[i + 1];

                    // Animate the CURRENT card shrinking/fading as the NEXT card overlaps it
                    gsap.to(card, {
                        scale: 0.9,
                        opacity: 0.5,
                        filter: "blur(5px)",
                        ease: "none",
                        scrollTrigger: {
                            trigger: nextCard,
                            start: "top bottom", // When top of next card hits bottom of viewport
                            end: "top top",      // When top of next card hits top of viewport
                            scrub: true
                        }
                    });
                });
            }
        });
    }
    // ===============================================
    // 9. NEW: EXPLODING TEXT EFFECT (TUNED)
    // ===============================================
    const explodeTitle = document.querySelector('[data-effect27]');
    if (explodeTitle && typeof gsap !== 'undefined') {
        const words = [...explodeTitle.querySelectorAll('.word')];

        gsap.fromTo(words,
            {
                // START STATE: Scattered, blurred, slightly transparent
                'will-change': 'opacity, transform',
                z: () => gsap.utils.random(100, 300), // Reduced depth so it's not in your face
                opacity: 0,
                xPercent: () => gsap.utils.random(-30, 30), // Reduced scatter width
                yPercent: () => gsap.utils.random(-30, 30), // Reduced scatter height
                rotationX: () => gsap.utils.random(-45, 45),
                rotationY: () => gsap.utils.random(-45, 45),
                filter: "blur(10px)"
            },
            {
                // END STATE: Clean, readable text
                ease: 'power3.out',
                opacity: 1,
                rotationX: 0,
                rotationY: 0,
                xPercent: 0,
                yPercent: 0,
                z: 0,
                filter: "blur(0px)",
                stagger: { each: 0.01, from: 'center' }, // Stagger from center looks better
                scrollTrigger: {
                    trigger: ".explode-text-section", // Trigger based on the section wrapper
                    start: "top top",
                    end: "bottom bottom",
                    scrub: 1.5, // Slower scrub for smoother feel
                }
            }
        );
    }

    /* 
   Ensure GSAP and ScrollTrigger are registered before running this.
   If using modules: 
   import gsap from "gsap";
   import { ScrollTrigger } from "gsap/ScrollTrigger";
   gsap.registerPlugin(ScrollTrigger);
*/

    // --- Utility: Preload Images ---
    const preloadImages = (selector = 'img') => {
        return new Promise((resolve) => {
            const images = document.querySelectorAll(selector);
            const total = images.length;
            if (total === 0) resolve();

            let loaded = 0;
            const onImageLoad = () => {
                loaded++;
                if (loaded === total) resolve();
            };

            images.forEach(img => {
                if (img.complete) {
                    onImageLoad();
                } else {
                    img.addEventListener('load', onImageLoad);
                    img.addEventListener('error', onImageLoad);
                }
            });
        });
    };

    // --- Animation Class ---
    class SectionAnimation {
        constructor() {
            this.dom = document.querySelector(".section");
            // Guard clause in case the element isn't found
            if (!this.dom) return;

            this.frontImages = this.dom.querySelectorAll(".section__media__front");
            this.smallImages = this.dom.querySelectorAll(".section__images img");
        }

        init() {
            if (!this.dom) return;

            this.timeline = gsap.timeline({
                scrollTrigger: {
                    trigger: this.dom,
                    start: "top top",
                    end: "bottom top",
                    scrub: true,
                    pin: true,
                    onUpdate: (self) => {
                        // Update CSS variable --progress for CSS-based animations (Text & Scale)
                        const easedProgress = gsap.parseEase("power1.inOut")(self.progress);
                        this.dom.style.setProperty("--progress", easedProgress);
                    }
                }
            });

            this.animate();
        }

        animate() {
            // Prepare 3D settings for small images
            gsap.set(this.smallImages, {
                transformStyle: "preserve-3d",
                backfaceVisibility: "hidden",
                force3D: true
            });

            // 1. Animate small images flying out
            this.timeline.to(this.smallImages, {
                z: "100vh",
                duration: 1,
                ease: "power1.inOut",
                stagger: {
                    amount: 0.2,
                    from: "center"
                }
            });

            // 2. Scale up front sliced images
            this.timeline.to(this.frontImages, {
                scale: 1,
                duration: 1,
                ease: "power1.inOut",
                delay: .1,
            }, 0.6); // Start time overlap

            // 3. Remove blur from front images
            this.timeline.to(this.frontImages, {
                duration: 1,
                filter: "blur(0px)",
                ease: "power1.inOut",
                delay: .4,
                stagger: {
                    amount: 0.2,
                    from: "end"
                }
            }, 0.6);
        }
    }

    // --- Initialization ---
    // Run this when your page loads
    const initScrollSection = () => {
        // Optional: Preload specific images within the section before starting
        preloadImages('.section img').then(() => {
            const animation = new SectionAnimation();
            animation.init();

            // Remove loading class if you have one on the body
            document.body.classList.remove('loading');
        });
    };

    // Start
    initScrollSection();

    // ===============================================
    // 10. ANALYSIS PAGE LOGIC (Upload & Camera)
    // ===============================================
    const uploadForm = document.getElementById('uploadForm');
    const resultsSection = document.getElementById('results-section');
    const downloadBtn = document.getElementById('download-report-btn');
    const btnUploadImage = document.getElementById('btn-upload-image');
    const btnUseCamera = document.getElementById('btn-use-camera');
    const uploadSection = document.getElementById('upload-section');
    const cameraSection = document.getElementById('camera-section');
    const fileInput = document.getElementById('file-input');
    const dropZone = document.getElementById('drop-zone');
    const previewContainer = document.getElementById('file-preview-container');
    const previewImg = document.getElementById('file-preview');
    const btnRemoveFile = document.getElementById('btn-remove-file');
    const btnChangeFile = document.getElementById('btn-change-file');
    const btnAnalyzeUpload = document.getElementById('btn-analyze-upload');
    const btnStartCamera = document.getElementById('btn-start-camera');
    const btnCapturePhoto = document.getElementById('btn-capture-photo');
    const btnRetakePhoto = document.getElementById('btn-retake-photo');
    const btnAnalyzeCapture = document.getElementById('btn-analyze-capture');
    const btnStopCamera = document.getElementById('btn-stop-camera');
    const video = document.getElementById('camera-feed');
    const captureCanvas = document.getElementById('capture-canvas');
    const cameraPlaceholder = document.getElementById('camera-placeholder');
    const loadingState = document.getElementById('loading-state');
    const loadingStatusText = document.getElementById('loading-status-text');

    if (uploadForm || btnUploadImage) {
        console.log('Analysis page initialized');

        let currentStream = null;
        let currentImageFile = null;

        updateDownloadButtonLabel();

        // UI Toggles
        if (btnUploadImage) {
            btnUploadImage.addEventListener('click', () => {
                hideError();
                if (cameraSection) cameraSection.classList.add('hidden');
                if (uploadSection) uploadSection.classList.remove('hidden');
                stopCameraStream();
                resetUploadState();
            });
        }

        if (btnUseCamera) {
            btnUseCamera.addEventListener('click', () => {
                hideError();
                if (uploadSection) uploadSection.classList.add('hidden');
                if (cameraSection) cameraSection.classList.remove('hidden');
                resetUploadState();
            });
        }

        // File Handling
        if (dropZone && fileInput) {
            dropZone.addEventListener('click', () => fileInput.click());

            dropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                dropZone.style.borderColor = '#007bff';
                dropZone.style.backgroundColor = '#f0f8ff';
            });

            dropZone.addEventListener('dragleave', () => {
                dropZone.style.borderColor = '#e0e0e0';
                dropZone.style.backgroundColor = 'transparent';
            });

            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZone.style.borderColor = '#e0e0e0';
                dropZone.style.backgroundColor = 'transparent';
                if (e.dataTransfer.files.length > 0) {
                    handleFileSelection(e.dataTransfer.files[0]);
                }
            });

            fileInput.addEventListener('change', () => {
                if (fileInput.files && fileInput.files.length > 0) {
                    handleFileSelection(fileInput.files[0]);
                }
            });
        }

        function handleFileSelection(file) {
            hideError();
            if (!file.type.startsWith('image/')) {
                alert('Please select an image file (JPG, PNG, etc.).');
                return;
            }
            if (file.size > 5 * 1024 * 1024) {
                alert('File size exceeds 5MB limit.');
                return;
            }

            currentImageFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                if (previewImg) previewImg.src = e.target.result;
                if (dropZone) dropZone.classList.add('hidden');
                if (previewContainer) previewContainer.classList.remove('hidden');
                if (btnAnalyzeUpload) btnAnalyzeUpload.disabled = false;
            };
            reader.readAsDataURL(file);
        }

        if (btnRemoveFile) btnRemoveFile.addEventListener('click', resetUploadState);
        if (btnChangeFile) btnChangeFile.addEventListener('click', () => fileInput && fileInput.click());

        function resetUploadState() {
            // Reset File Upload UI
            if (fileInput) fileInput.value = '';
            if (previewContainer) previewContainer.classList.add('hidden');
            if (dropZone) dropZone.classList.remove('hidden');
            if (btnAnalyzeUpload) btnAnalyzeUpload.disabled = true;
            currentImageFile = null;

            // Reset Camera UI
            if (typeof stopCameraStream === 'function') {
                stopCameraStream();
            }
            if (captureCanvas) captureCanvas.classList.add('hidden');
            if (cameraPlaceholder) cameraPlaceholder.classList.remove('hidden');
            if (btnStartCamera) btnStartCamera.classList.remove('hidden');
            if (btnAnalyzeCapture) btnAnalyzeCapture.classList.add('hidden');
            if (btnRetakePhoto) btnRetakePhoto.classList.add('hidden');
        }

        // Upload Analysis
        if (uploadForm) {
            uploadForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                if (!currentImageFile && (!fileInput || !fileInput.files || fileInput.files.length === 0)) {
                    alert('Please select an image file first.');
                    return;
                }
                const fileToUpload = currentImageFile || fileInput.files[0];
                const formData = new FormData();
                formData.append('file', fileToUpload);
                showLoading();
                await analyzeImage(formData);
            });
        }

        // Camera Logic
        if (btnStartCamera) btnStartCamera.addEventListener('click', startCamera);
        if (btnCapturePhoto) btnCapturePhoto.addEventListener('click', capturePhoto);
        if (btnRetakePhoto) btnRetakePhoto.addEventListener('click', retakePhoto);
        if (btnStopCamera) btnStopCamera.addEventListener('click', stopCameraStream);

        if (btnAnalyzeCapture) {
            btnAnalyzeCapture.addEventListener('click', async () => {
                if (!captureCanvas) {
                    alert('Camera system error.');
                    return;
                }
                captureCanvas.toBlob(async (blob) => {
                    if (!blob) {
                        alert('Failed to process captured image.');
                        return;
                    }
                    const formData = new FormData();
                    formData.append('file', blob, 'captured_image.jpg');
                    showLoading();
                    await analyzeImage(formData);
                }, 'image/jpeg', 0.9);
            });
        }

        function startCamera() {
            try {
                console.log('Attempting to start camera...');
                if (typeof hideError === 'function') hideError();

                if (!video) {
                    console.error('Video element not found');
                    alert('Camera interface error (Video Element Missing)');
                    return;
                }
                if (currentStream) {
                    console.log('Stream already active');
                    return;
                }

                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                    alert('Camera API not supported in this browser/context');
                    return;
                }

                navigator.mediaDevices.getUserMedia({
                    video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
                })
                    .then(stream => {
                        console.log('Camera stream acquired');
                        currentStream = stream;
                        video.srcObject = stream;
                        video.classList.remove('hidden');
                        if (captureCanvas) captureCanvas.classList.add('hidden');
                        if (cameraPlaceholder) cameraPlaceholder.classList.add('hidden');

                        if (btnStartCamera) btnStartCamera.classList.add('hidden');
                        if (btnCapturePhoto) btnCapturePhoto.classList.remove('hidden');
                        if (btnStopCamera) btnStopCamera.classList.remove('hidden');
                        if (btnRetakePhoto) btnRetakePhoto.classList.add('hidden');
                        if (btnAnalyzeCapture) btnAnalyzeCapture.classList.add('hidden');
                    })
                    .catch(error => {
                        console.error('Camera error:', error);
                        alert('Unable to access camera. Please check permissions. Error: ' + error.message);
                    });
            } catch (e) {
                console.error('Critical startCamera error:', e);
                alert('Critical Camera Error: ' + e.message);
            }
        }

        function capturePhoto() {
            if (!video || !captureCanvas || !currentStream) return;
            if (!video.videoWidth) return;

            captureCanvas.width = video.videoWidth;
            captureCanvas.height = video.videoHeight;
            const ctx = captureCanvas.getContext('2d');
            ctx.drawImage(video, 0, 0);

            video.classList.add('hidden');
            captureCanvas.classList.remove('hidden');

            if (btnCapturePhoto) btnCapturePhoto.classList.add('hidden');
            if (btnRetakePhoto) btnRetakePhoto.classList.remove('hidden');
            if (btnAnalyzeCapture) btnAnalyzeCapture.classList.remove('hidden');
            if (btnStopCamera) btnStopCamera.classList.add('hidden');
        }

        function retakePhoto() {
            if (video && currentStream) {
                video.srcObject = currentStream;
                video.classList.remove('hidden');
            }
            if (captureCanvas) captureCanvas.classList.add('hidden');

            if (btnCapturePhoto) btnCapturePhoto.classList.remove('hidden');
            if (btnRetakePhoto) btnRetakePhoto.classList.add('hidden');
            if (btnAnalyzeCapture) btnAnalyzeCapture.classList.add('hidden');
            if (btnStopCamera) btnStopCamera.classList.remove('hidden');
        }

        function stopCameraStream() {
            if (currentStream) {
                currentStream.getTracks().forEach(track => track.stop());
                currentStream = null;
            }
            if (video) {
                video.srcObject = null;
                video.classList.add('hidden');
            }
            if (captureCanvas) captureCanvas.classList.add('hidden');
            if (cameraPlaceholder) cameraPlaceholder.classList.remove('hidden');

            if (btnStartCamera) btnStartCamera.classList.remove('hidden');
            if (btnCapturePhoto) btnCapturePhoto.classList.add('hidden');
            if (btnRetakePhoto) btnRetakePhoto.classList.add('hidden');
            if (btnAnalyzeCapture) btnAnalyzeCapture.classList.add('hidden');
            if (btnStopCamera) btnStopCamera.classList.add('hidden');
        }

        // Analysis API Call
        async function analyzeImage(formData) {
            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });

                if (!response.ok) {
                    const errorText = await response.text();
                    try {
                        const errData = JSON.parse(errorText);
                        throw new Error(errData.error || errorText || 'Server error');
                    } catch (e) {
                        // If JSON parse fails, throw the raw text
                        if (e instanceof SyntaxError) throw new Error(errorText);
                        throw e;
                    }
                }

                const data = await response.json();
                if (data.error) throw new Error(data.error);

                completeLoadingAnimation();
                hideLoading();
                displayResults(data);
            } catch (err) {
                console.error(err);

                // Parse clean message if possible
                let msg = err.message;
                try {
                    // Try parsing as JSON first
                    const parsed = JSON.parse(msg);
                    if (parsed.error) msg = parsed.error;
                } catch (e) {
                    // Fallback to Regex if JSON parse fails but looks like JSON
                    const match = msg.match(/"error"\s*:\s*"(.*?)"/);
                    if (match && match[1]) msg = match[1];
                }

                showError(msg);
                hideLoading();

                // Clear the invalid image so the user knows it was rejected
                resetUploadState();
            }
        }

        function showError(msg) {
            const banner = document.getElementById('error-banner');
            const errorText = document.getElementById('error-text');
            if (banner && errorText) {
                errorText.innerText = msg;
                banner.classList.remove('hidden');
            } else {
                alert('Analysis failed: ' + msg);
            }
        }

        function hideError() {
            const banner = document.getElementById('error-banner');
            if (banner) banner.classList.add('hidden');
        }

        function showLoading() {
            document.querySelector('.clean-card')?.classList.add('hidden');
            if (resultsSection) resultsSection.classList.add('hidden');
            if (loadingState) {
                loadingState.classList.remove('hidden');
                loadingState.classList.remove('processing-complete');
                if (loadingStatusText) loadingStatusText.textContent = 'Uploading image securely...';
            }
        }

        function completeLoadingAnimation() {
            if (loadingStatusText) loadingStatusText.textContent = 'Generating AI insights...';
            if (loadingState) loadingState.classList.add('processing-complete');
        }

        function hideLoading() {
            if (loadingState) loadingState.classList.add('hidden');
            document.querySelector('.clean-card')?.classList.remove('hidden');
        }

        window.addEventListener('beforeunload', stopCameraStream);
    }

    // ===============================================
    // 11. RESULTS & REPORT
    // ===============================================
    let currentResultData = null;

    function displayResults(data) {
        currentResultData = data;
        const resultsSection = document.getElementById('results-section');
        if (!resultsSection) return;

        resultsSection.classList.remove('hidden');

        const confidenceBadge = document.getElementById('confidence-badge');
        if (confidenceBadge) confidenceBadge.innerText = `${(data.confidence * 100).toFixed(1)}% CONFIDENCE`;

        const statusEl = document.getElementById('prediction-status');
        if (statusEl) {
            let predictionText = data.prediction;
            if (['Non-Cancerous', 'Non Cancerous', 'Healthy'].includes(predictionText)) {
                statusEl.innerText = 'NON-CANCEROUS';
                statusEl.className = 'prediction-status healthy';
            } else {
                statusEl.innerText = `${predictionText.toUpperCase()} DETECTED`;
                statusEl.className = `prediction-status ${predictionText.includes('Cancer') ? 'cancer' : 'healthy'}`;
            }
        }

        const origImg = document.getElementById('result-orig-img');
        if (origImg && data.filename) origImg.src = `/static/uploads/${data.filename}`;

        const gradcamImg = document.getElementById('result-gradcam-img');
        if (gradcamImg && data.gradcam) gradcamImg.src = `/static/gradcam/${data.gradcam}`;

        const hospitalBtn = document.getElementById('find-hospitals-btn');
        if (hospitalBtn) {
            const isCancer = data.prediction.toLowerCase().includes('cancer');
            hospitalBtn.classList.toggle('hidden', !isCancer);
        }

        saveToHistory(data);

        try {
            const payload = { ...data, timestamp: new Date().toISOString() };
            sessionStorage.setItem('pendingResultData', JSON.stringify(payload));
            setCookie('oro_last_analysis', JSON.stringify(payload));
        } catch (e) { console.warn(e); }
    }

    if (downloadBtn) {
        downloadBtn.addEventListener('click', async () => {
            if (!currentResultData) return alert('No analysis results available.');
            userAuthenticated = window.__IS_AUTHENTICATED === true || window.__IS_AUTHENTICATED === 'true';

            if (!userAuthenticated) {
                sessionStorage.setItem('pendingDownload', 'true');
                sessionStorage.setItem('pendingReportData', JSON.stringify(currentResultData));
                sessionStorage.setItem('pendingResultData', JSON.stringify(currentResultData));
                sessionStorage.setItem('pendingRoute', window.location.pathname);
                return window.location.href = `/login?next=${encodeURIComponent('/analyze')}`;
            }

            downloadBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Generating...';
            downloadBtn.disabled = true;
            try {
                const response = await fetch('/download_report', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(currentResultData)
                });
                if (!response.ok) throw new Error(await response.text());

                const contentDisposition = response.headers.get('Content-Disposition');
                let filename = `OroEYE_Report_${new Date().getTime()}.pdf`;

                if (contentDisposition && contentDisposition.indexOf('filename=') !== -1) {
                    filename = contentDisposition.split('filename=')[1].replace(/"/g, '');
                }

                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                a.type = 'application/pdf';
                document.body.appendChild(a);
                a.click();
                setTimeout(() => {
                    document.body.removeChild(a);
                    window.URL.revokeObjectURL(url);
                }, 100);
            } catch (err) {
                alert('Download failed: ' + err.message);
            } finally {
                updateDownloadButtonLabel();
                downloadBtn.disabled = false;
            }
        });
    }

    // State Restoration (e.g. after login)
    const pendingDownload = sessionStorage.getItem('pendingDownload');
    const pendingReportData = sessionStorage.getItem('pendingReportData');
    const pendingResultData = sessionStorage.getItem('pendingResultData');

    if (pendingResultData && document.getElementById('results-section')?.classList.contains('hidden')) {
        try {
            const restoredData = JSON.parse(pendingResultData);
            setTimeout(() => {
                document.querySelector('.clean-card')?.classList.add('hidden');
                displayResults(restoredData);
            }, 100);
        } catch (err) { }
    }

    if (pendingDownload === 'true' && pendingReportData) {
        sessionStorage.removeItem('pendingDownload');
        try {
            currentResultData = JSON.parse(pendingReportData);
            setTimeout(() => document.getElementById('download-report-btn')?.click(), 1200);
            sessionStorage.removeItem('pendingReportData');
        } catch (e) { }
    }

    // Reset Button
    const resetBtn = document.getElementById('reset-analysis-btn');
    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            document.getElementById('results-section')?.classList.add('hidden');
            document.getElementById('loading-state')?.classList.add('hidden');
            document.querySelector('.clean-card')?.classList.remove('hidden');
            document.getElementById('upload-section')?.classList.remove('hidden');
            document.getElementById('camera-section')?.classList.add('hidden');
            if (fileInput) fileInput.value = '';
            if (previewContainer) previewContainer.classList.add('hidden');
            if (dropZone) dropZone.classList.remove('hidden');
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }

    // Find Hospital Button
    const hospitalBtn = document.getElementById('find-hospitals-btn');
    if (hospitalBtn) {
        hospitalBtn.addEventListener('click', () => {
            const query = 'oncologist oral cancer clinic hospital';
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(
                    (pos) => window.open(`https://www.google.com/maps/search/${query}/@${pos.coords.latitude},${pos.coords.longitude},13z`, '_blank'),
                    () => window.open(`https://www.google.com/maps/search/${query}`, '_blank')
                );
            } else {
                window.open(`https://www.google.com/maps/search/${query}`, '_blank');
            }
        });
    }

    // History Logic
    const historySection = document.getElementById('history-section');
    const historyGrid = document.getElementById('history-grid');
    const clearHistoryBtn = document.getElementById('clear-history-btn');

    if (historySection && historyGrid) {
        loadHistory();
        if (clearHistoryBtn) {
            clearHistoryBtn.addEventListener('click', () => {
                loadHistory();
            });
        }
    }

    function saveToHistory(data) { loadHistory(); }

    async function loadHistory() {
        if (!userAuthenticated || !historyGrid) {
            historySection?.classList.add('hidden');
            return;
        }
        try {
            const response = await fetch('/api/history');
            const history = response.ok ? await response.json() : [];
            renderHistory(history);
        } catch (e) { renderHistory([]); }
    }

    function renderHistory(history) {
        if (history.length > 0) {
            historySection?.classList.remove('hidden');
            historyGrid.innerHTML = history.map(item => `
                <div class="clean-card" style="padding: 1.5rem;">
                    <div style="color: #888; font-size: 0.8rem; margin-bottom: 0.5rem;">${new Date(item.timestamp).toLocaleString()}</div>
                    <div style="font-weight: 700; text-transform: uppercase; color: ${item.prediction === 'Cancer' ? 'var(--danger)' : 'var(--success)'}">
                        ${item.prediction.toUpperCase()}
                    </div>
                    <div style="font-size: 0.9rem;">Confidence: ${(item.confidence * 100).toFixed(1)}%</div>
                </div>`).join('');
        } else {
            historySection?.classList.add('hidden');
            historyGrid.innerHTML = '';
        }
    }

    function updateDownloadButtonLabel() {
        if (!downloadBtn) return;
        const text = window.__IS_AUTHENTICATED ? 'Download Report' : 'Download Report (Login Required)';
        downloadBtn.innerHTML = `<i class="fa-solid fa-file-pdf" style="margin-right: 0.5rem;"></i> ${text}`;
    }

    function setCookie(name, value, days = 7) {
        const expires = new Date(Date.now() + days * 24 * 60 * 60 * 1000).toUTCString();
        document.cookie = `${name}=${encodeURIComponent(value)}; expires=${expires}; path=/`;
    }
});


// ===============================================
// 12. RETRO CAMERA FOOTER LOGIC (DEFINITIVE FINAL)
// ===============================================
const retroFooter = document.getElementById('retro-camera-footer');
if (retroFooter && typeof gsap !== 'undefined' && typeof Flip !== 'undefined') {
    const eyes = retroFooter.querySelectorAll('.eye-o');
    const spotlight = retroFooter.querySelector('#spotlight');
    const cameraScene = retroFooter.querySelector('#camera-scene');
    const logoStack = retroFooter.querySelector('#logo-text');
    const video = retroFooter.querySelector('#webcam');
    const shutter = retroFooter.querySelector('#trigger');
    const flashBurst = retroFooter.querySelector('.flash-burst');
    const globalFlash = document.getElementById('global-flash');
    const canvas = retroFooter.querySelector('#proc-canvas');
    const powerBtn = retroFooter.querySelector('#power-btn');
    const stage1Container = document.getElementById('photo-stage-1');
    const stage2Container = document.getElementById('photo-stage-2');
    const apertureBlades = retroFooter.querySelectorAll('.aperture-blade');

    let isBusy = false, isCameraOn = false, stream = null, lastPhotoDataUrl = null;

    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    function sfx(type) {
        const t = audioCtx.currentTime; const o = audioCtx.createOscillator(); const g = audioCtx.createGain(); o.connect(g); g.connect(audioCtx.destination);
        if (type === 'shutter') { o.type = 'square'; o.frequency.setValueAtTime(400, t); o.frequency.exponentialRampToValueAtTime(100, t + 0.1); g.gain.setValueAtTime(0.3, t); g.gain.exponentialRampToValueAtTime(0.01, t + 0.1); o.start(t); o.stop(t + 0.1); }
        else if (type === 'eject') { o.type = 'sawtooth'; o.frequency.setValueAtTime(150, t); o.frequency.exponentialRampToValueAtTime(50, t + 0.5); g.gain.setValueAtTime(0.1, t); g.gain.linearRampToValueAtTime(0, t + 0.5); o.start(t); o.stop(t + 0.5); }
    }

    async function startVideo() { if (stream) return; try { stream = await navigator.mediaDevices.getUserMedia({ video: { width: 800, height: 800, facingMode: 'user' } }); video.srcObject = stream; isCameraOn = true; powerBtn.classList.add('active'); } catch (e) { console.error("Camera error:", e); } }
    function stopVideo() { if (stream) { stream.getTracks().forEach(track => track.stop()); stream = null; video.srcObject = null; } isCameraOn = false; powerBtn.classList.remove('active'); }
    function toggleCamera(force) { (force ?? !isCameraOn) ? startVideo() : stopVideo(); }

    powerBtn.addEventListener('click', () => toggleCamera());
    new IntersectionObserver(e => e.forEach(entry => toggleCamera(entry.isIntersecting)), { threshold: 0.1 }).observe(retroFooter);

    document.addEventListener('mousemove', e => {
        if (retroFooter.getBoundingClientRect().top > window.innerHeight) return;
        const { clientX, clientY } = e;
        const spotX = (clientX / window.innerWidth - 0.5) * -80;
        const spotY = (clientY / window.innerHeight - 0.5) * -80;
        spotlight.style.transform = `translate(calc(-50% + ${spotX}px), calc(-50% + ${spotY}px))`;
        const tiltX = (clientX / window.innerWidth - 0.5) * 10;
        const tiltY = (clientY / window.innerHeight - 0.5) * -10;
        logoStack.style.transform = `rotateY(${tiltX}deg) rotateX(${tiltY}deg)`;
        if (!e.target.closest('.power-btn, .shutter')) {
            cameraScene.style.transform = `rotateY(${(clientX / window.innerWidth - 0.5) * 20}deg) rotateX(${(clientY / window.innerHeight - 0.5) * -15}deg)`;
        }

        eyes.forEach(eye => {
            const pupil = eye.querySelector('.pupil');
            const sclera = eye.querySelector('.sclera');
            const rect = sclera.getBoundingClientRect();
            const angle = Math.atan2(clientY - (rect.top + rect.height / 2), clientX - (rect.left + rect.width / 2));
            const maxRadX = (rect.width - pupil.offsetWidth) / 2 * 0.9;
            const maxRadY = (rect.height - pupil.offsetHeight) / 2 * 0.9;
            gsap.to(pupil, { x: Math.cos(angle) * maxRadX, y: Math.sin(angle) * maxRadY, duration: 0.1, ease: 'power1.out' });
        });
    });

    shutter.addEventListener('click', () => {
        if (isBusy || !isCameraOn) return;
        isBusy = true; shutter.classList.add('disabled'); stage1Container.innerHTML = ''; stage2Container.innerHTML = '';
        sfx('shutter'); flashBurst.classList.add('active'); if (globalFlash) globalFlash.classList.add('active');
        gsap.timeline().to(apertureBlades, { scale: 1, duration: 0.05, ease: 'power2.out', stagger: 0.01 }).to(apertureBlades, { scale: 0, duration: 0.2, ease: 'power2.in', delay: 0.05 });

        setTimeout(() => {
            flashBurst.classList.remove('active'); if (globalFlash) globalFlash.classList.remove('active');
            const ctx = canvas.getContext('2d'); canvas.width = 800; canvas.height = 800; ctx.translate(800, 0); ctx.scale(-1, 1); ctx.drawImage(video, 0, 0, 800, 800); ctx.setTransform(1, 0, 0, 1, 0, 0); lastPhotoDataUrl = canvas.toDataURL('image/jpeg', 0.9);
            if (window.innerWidth > 900) {
                const photo = document.createElement('div'); photo.className = 'photo-card'; const photoImg = document.createElement('img'); photoImg.src = lastPhotoDataUrl; photo.appendChild(photoImg); stage1Container.appendChild(photo);
                setTimeout(() => { sfx('eject'); photo.classList.add('visible'); cameraScene.classList.add('recoiling'); setTimeout(() => photoImg.classList.add('developed'), 100); setTimeout(() => cameraScene.classList.remove('recoiling'), 200); isBusy = false; shutter.classList.remove('disabled'); }, 50);
                photo.addEventListener('click', handlePhotoClick, { once: true });
            } else { sfx('eject'); handlePhotoClick(null); }
        }, 150);
    });

    function handlePhotoClick(event) {
        const photo = event ? event.currentTarget : null;
        isBusy = true;
        stage2Container.innerHTML = `<div class="photo-stage-2-container"><div class="photo-card" data-flip-id="photo"><img src="${lastPhotoDataUrl}" class="developing"></div><a href="#" class="dl-btn">Download Image</a></div>`;
        const downloadBtn = stage2Container.querySelector('.dl-btn');
        const finalImageEl = stage2Container.querySelector('img');
        setTimeout(() => finalImageEl.classList.add('developed'), 50);

        const compCanvas = document.createElement('canvas'); compCanvas.width = 1000, compCanvas.height = 1200; const cCtx = compCanvas.getContext("2d"); cCtx.fillStyle = "#fff", cCtx.fillRect(0, 0, 1000, 1200); const img = new Image; img.onload = function () { cCtx.drawImage(img, 50, 50, 900, 900), cCtx.fillStyle = "#444", cCtx.font = "bold 80px 'Italiana', serif", cCtx.textAlign = "center", cCtx.textBaseline = "middle", cCtx.fillText("OrOEYE", 500, 1075), downloadBtn.href = compCanvas.toDataURL("image/jpeg", .9), downloadBtn.download = `ORO_EYE_${Date.now()}.jpg` }, img.src = lastPhotoDataUrl;

        if (photo) {
            const state = Flip.getState(photo);
            stage2Container.querySelector(".photo-stage-2-container").prepend(photo);
            Flip.from(state, { duration: 0.6, ease: "power2.inOut", onComplete: () => { downloadBtn.classList.add('visible'); photo.remove(); isBusy = false; } });
        } else {
            const finalCard = stage2Container.querySelector('.photo-card');
            gsap.from(finalCard, { opacity: 0, scale: 0.8, duration: 0.6, ease: 'power2.out', onComplete: () => { downloadBtn.classList.add('visible'); isBusy = false; } });
        }
    }
}