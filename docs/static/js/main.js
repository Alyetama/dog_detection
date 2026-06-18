(function () {
  "use strict";

  // Theme toggle
  const html = document.documentElement;
  const themeToggle = document.getElementById("theme-toggle");
  const storedTheme = localStorage.getItem("theme");

  function applyTheme(theme) {
    html.setAttribute("data-theme", theme);
    if (themeToggle) {
      themeToggle.textContent = theme === "dark" ? "☀️" : "🌙";
    }
  }

  if (storedTheme) {
    applyTheme(storedTheme);
  } else {
    const prefersLight = window.matchMedia("(prefers-color-scheme: light)").matches;
    applyTheme(prefersLight ? "light" : "dark");
  }

  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      const current = html.getAttribute("data-theme") || "dark";
      const next = current === "dark" ? "light" : "dark";
      applyTheme(next);
      localStorage.setItem("theme", next);
    });
  }

  // Mobile menu
  const mobileToggle = document.querySelector(".mobile-menu-toggle");
  const navLinks = document.querySelector(".nav-links");
  if (mobileToggle && navLinks) {
    mobileToggle.addEventListener("click", () => {
      navLinks.classList.toggle("open");
    });
  }

  // Copy buttons
  document.querySelectorAll("[data-copy]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const code = btn.parentElement.querySelector("pre")?.innerText || "";
      navigator.clipboard.writeText(code).then(() => {
        const original = btn.textContent;
        btn.textContent = "Copied!";
        btn.classList.add("copied");
        setTimeout(() => {
          btn.textContent = original;
          btn.classList.remove("copied");
        }, 1500);
      });
    });
  });

  // Demo page logic
  const dropZone = document.getElementById("drop-zone");
  const imageInput = document.getElementById("image-input");
  const uploadPreview = document.getElementById("upload-preview");
  const previewImg = document.getElementById("preview-img");
  const removeBtn = document.getElementById("remove-btn");
  const detectBtn = document.getElementById("detect-btn");
  const resultPanel = document.getElementById("result-panel");
  const resultPlaceholder = document.getElementById("result-placeholder");
  const resultContent = document.getElementById("result-content");
  const resultImg = document.getElementById("result-img");
  const resultMeta = document.getElementById("result-meta");
  const resultError = document.getElementById("result-error");
  const errorMessage = document.getElementById("error-message");

  const confInput = document.getElementById("conf");
  const iouInput = document.getElementById("iou");
  const imgszInput = document.getElementById("imgsz");
  const confValue = document.getElementById("conf-value");
  const iouValue = document.getElementById("iou-value");

  if (dropZone) {
    function updateRangeDisplay(input, display) {
      if (!input || !display) return;
      display.textContent = parseFloat(input.value).toFixed(
        input.step.includes(".") ? 2 : 0
      );
    }

    if (confInput && confValue) {
      confInput.addEventListener("input", () => updateRangeDisplay(confInput, confValue));
      updateRangeDisplay(confInput, confValue);
    }
    if (iouInput && iouValue) {
      iouInput.addEventListener("input", () => updateRangeDisplay(iouInput, iouValue));
      updateRangeDisplay(iouInput, iouValue);
    }

    let selectedFile = null;

    function resetResultPanel() {
      resultPlaceholder.innerHTML = `
        <div class="placeholder-icon">🐕</div>
        <p>Your annotated result will appear here</p>
      `;
      resultPlaceholder.hidden = false;
      resultContent.hidden = true;
      resultError.hidden = true;
    }

    function setFile(file) {
      if (!file || !file.type.startsWith("image/")) return;
      selectedFile = file;
      const url = URL.createObjectURL(file);
      previewImg.src = url;
      uploadPreview.hidden = false;
      detectBtn.disabled = false;
      resetResultPanel();
    }

    function clearFile() {
      selectedFile = null;
      uploadPreview.hidden = true;
      previewImg.src = "";
      detectBtn.disabled = true;
      imageInput.value = "";
      resetResultPanel();
    }

    dropZone.addEventListener("click", () => imageInput.click());

    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", () => {
      dropZone.classList.remove("drag-over");
    });

    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("drag-over");
      const file = e.dataTransfer.files[0];
      setFile(file);
    });

    imageInput.addEventListener("change", () => {
      const file = imageInput.files[0];
      setFile(file);
    });

    removeBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      clearFile();
    });

    detectBtn.addEventListener("click", async () => {
      if (!selectedFile) return;

      const btnText = detectBtn.querySelector(".btn-text");
      const spinner = detectBtn.querySelector(".spinner");
      detectBtn.disabled = true;
      btnText.textContent = "Running inference...";
      spinner.hidden = false;
      resultPlaceholder.hidden = true;
      resultContent.hidden = true;
      resultError.hidden = true;

      const formData = new FormData();
      formData.append("image", selectedFile);
      formData.append("conf", confInput.value);
      formData.append("iou", iouInput.value);
      formData.append("imgsz", imgszInput.value);

      try {
        const response = await fetch("/predict", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          const data = await response.json().catch(() => ({}));
          throw new Error(data.detail || `Inference failed (${response.status})`);
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        resultImg.src = url;

        const detectionsCount = parseInt(response.headers.get("X-Detections") || "0", 10);
        const inferenceTime = response.headers.get("X-Inference-Time-Ms") || "—";
        let detections = [];
        try {
          const raw = response.headers.get("X-Detections-Json");
          if (raw) detections = JSON.parse(raw);
        } catch {
          detections = [];
        }

        resultError.hidden = true;

        if (detectionsCount === 0) {
          resultContent.hidden = true;
          resultPlaceholder.innerHTML = `
            <div class="placeholder-icon">🔍</div>
            <p>No dogs detected in this image.</p>
            <p class="upload-hint">Try lowering the confidence threshold or use a different image.</p>
          `;
          resultPlaceholder.hidden = false;
        } else {
          resultPlaceholder.hidden = true;
          const detectionLines = detections
            .map(
              (det) => `
                <div class="meta-item result-line">
                  <span class="meta-label">conf</span>
                  <span class="meta-value">${det.conf.toFixed(3)}</span>
                </div>
              `
            )
            .join("");
          resultMeta.innerHTML = `
            <div class="meta-item result-line">
              <span class="meta-label">Inference time:</span>
              <span class="meta-value">${inferenceTime} ms</span>
            </div>
            ${detectionLines}
          `;
          resultContent.hidden = false;
        }
      } catch (err) {
        errorMessage.textContent = err.message;
        resultError.hidden = false;
      } finally {
        detectBtn.disabled = false;
        btnText.textContent = "Run Detection";
        spinner.hidden = true;
      }
    });
  }

  // Lightbox for training results gallery
  const lightbox = document.getElementById("lightbox");
  if (lightbox) {
    const lightboxImg = document.getElementById("lightbox-img");
    const lightboxCaption = document.getElementById("lightbox-caption");
    const lightboxCounter = document.getElementById("lightbox-counter");
    const lightboxClose = document.getElementById("lightbox-close");
    const lightboxPrev = document.getElementById("lightbox-prev");
    const lightboxNext = document.getElementById("lightbox-next");
    const galleryItems = Array.from(document.querySelectorAll("[data-lightbox]"));

    let currentGroup = [];
    let currentIndex = 0;

    function buildGroups() {
      const groups = {};
      galleryItems.forEach((item) => {
        const groupName = item.dataset.lightboxGroup;
        if (!groups[groupName]) groups[groupName] = [];
        groups[groupName].push(item);
      });
      // Sort each group by index
      Object.keys(groups).forEach((name) => {
        groups[name].sort((a, b) => parseInt(a.dataset.lightboxIndex, 10) - parseInt(b.dataset.lightboxIndex, 10));
      });
      return groups;
    }

    const groups = buildGroups();

    function openLightbox(groupName, index) {
      currentGroup = groups[groupName] || [];
      currentIndex = index;
      showImage();
      lightbox.hidden = false;
      document.body.style.overflow = "hidden";
    }

    function showImage() {
      if (currentGroup.length === 0) return;
      const item = currentGroup[currentIndex];
      const img = item.querySelector("img");
      const caption = item.querySelector("figcaption");
      lightboxImg.src = img.src;
      lightboxImg.alt = img.alt;
      lightboxCaption.textContent = caption ? caption.textContent : "";
      lightboxCounter.textContent = currentGroup.length > 1 ? `${currentIndex + 1} / ${currentGroup.length}` : "";
      lightboxPrev.disabled = currentIndex === 0;
      lightboxNext.disabled = currentIndex === currentGroup.length - 1;
    }

    function closeLightbox() {
      lightbox.hidden = true;
      lightboxImg.src = "";
      document.body.style.overflow = "";
    }

    function nextImage() {
      if (currentIndex < currentGroup.length - 1) {
        currentIndex++;
        showImage();
      }
    }

    function prevImage() {
      if (currentIndex > 0) {
        currentIndex--;
        showImage();
      }
    }

    galleryItems.forEach((item) => {
      item.addEventListener("click", () => {
        const groupName = item.dataset.lightboxGroup;
        const index = parseInt(item.dataset.lightboxIndex, 10);
        openLightbox(groupName, index);
      });
    });

    lightboxClose.addEventListener("click", closeLightbox);
    lightboxNext.addEventListener("click", nextImage);
    lightboxPrev.addEventListener("click", prevImage);

    lightbox.addEventListener("click", (e) => {
      if (e.target === lightbox) closeLightbox();
    });

    document.addEventListener("keydown", (e) => {
      if (lightbox.hidden) return;
      if (e.key === "Escape") closeLightbox();
      if (e.key === "ArrowRight") nextImage();
      if (e.key === "ArrowLeft") prevImage();
    });
  }
})();
