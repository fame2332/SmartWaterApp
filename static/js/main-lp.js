const menuBtn = document.getElementById("menu-btn");
const navLinks = document.getElementById("nav-links");
const menuBtnIcon = menuBtn.querySelector("i"); 

menuBtn.addEventListener("click", () => {
    navLinks.classList.toggle("open");

    const isOpen = navLinks.classList.contains("open");
    menuBtnIcon.setAttribute("class", isOpen ? "bx bx-x" : "bx bx-menu");
});

navLinks.addEventListener("click", (e) => {
    navLinks.classList.remove("open");
    menuBtnIcon.setAttribute("class", "bx bx-menu")
});

const scrollRevealOption = {
    distance: "50px",
    origin: "bottom",
    duration: 1000,
};

ScrollReveal().reveal(".container-left h1", {
    ...scrollRevealOption,
});
ScrollReveal().reveal(".container-left .container-btn", {
    ...scrollRevealOption,
    delay: 500,
});

ScrollReveal().reveal(".container-right h4", {
    ...scrollRevealOption,
    delay: 2000,
});

ScrollReveal().reveal(".container-right h2", {
    ...scrollRevealOption,
    delay: 2500,
});
ScrollReveal().reveal(".container-right h3", {
    ...scrollRevealOption,
    delay: 3000,
});
ScrollReveal().reveal(".container-right p", {
    ...scrollRevealOption,
    delay: 3500,
});

ScrollReveal().reveal(".container-right .rain-1", {
    duration: 1000,
    delay: 4000,
});
ScrollReveal().reveal(".container-right .technology", {
    duration: 1000,
    delay: 4500,
});

ScrollReveal().reveal(".location", {
    ...scrollRevealOption,
    origin: "left",
    delay: 5000,
});

ScrollReveal().reveal(".location", {
    ...scrollRevealOption,
    origin: "left",
    delay: 5000,
});