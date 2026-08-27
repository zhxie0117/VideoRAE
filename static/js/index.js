$(document).ready(function() {
  $(".navbar-burger").click(function() {
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");
  });

  const heroWall = document.querySelector(".hero-wall");
  const heroVideos = [...document.querySelectorAll(".hero-wall-video")];
  const pageVideos = [...document.querySelectorAll(".example-gen-video")];
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const ensurePlaying = (video, offsetSeconds) => {
    if (!video || reducedMotion) return;
    video.muted = true;
    video.defaultMuted = true;
    const playWithOffset = () => {
      if (Number.isFinite(offsetSeconds) && offsetSeconds > 0 && Number.isFinite(video.duration) && video.duration > 0) {
        video.currentTime = offsetSeconds % video.duration;
      }
      video.play().catch(() => {});
    };
    if (video.readyState >= 1) playWithOffset();
    else video.addEventListener("loadedmetadata", playWithOffset, { once: true });
  };

  heroVideos.forEach((video, index) => {
    // Stagger playback so the wall feels denser / more "alive".
    ensurePlaying(video, (index * 0.37) % 2.4);
  });

  const updateNavbar = () => {
    const scrolled = heroWall
      ? heroWall.getBoundingClientRect().bottom <= 72
      : window.scrollY > 80;
    document.body.classList.toggle("navbar-scrolled", scrolled);
  };

  window.addEventListener("scroll", updateNavbar, { passive: true });
  window.addEventListener("resize", updateNavbar);
  updateNavbar();

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      heroVideos.forEach((video) => video.pause());
      return;
    }
    heroVideos.forEach((video) => ensurePlaying(video));
  });

  if ("IntersectionObserver" in window && pageVideos.length) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(({ target, isIntersecting }) => {
        if (isIntersecting) ensurePlaying(target);
        else if (!target.paused) target.pause();
      });
    }, { rootMargin: "120px 0px", threshold: 0.08 });

    pageVideos.forEach((video) => observer.observe(video));
  } else {
    pageVideos.forEach((video) => ensurePlaying(video));
  }

  const copyButton = document.querySelector("#copy-bibtex");
  const bibtex = document.querySelector("#bibtex-code");
  copyButton?.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(bibtex.textContent.trim());
      copyButton.textContent = "Copied";
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(bibtex);
      selection.removeAllRanges();
      selection.addRange(range);
      copyButton.textContent = "Selected";
    }
    window.setTimeout(() => { copyButton.textContent = "Copy"; }, 1800);
  });
});
