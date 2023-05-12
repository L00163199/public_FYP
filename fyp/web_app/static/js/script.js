/*

  Student Name: Dylan Walsh
  Student Number: L00163199
  Description: The purpose of this file is to
  provide some javascript functions required for
  the web app.
  Note: Though not good practice, some internal javascript
  has been used to cater to file-specific requirements
  in this project.

*/

// The below functions add responsiveness
// and animation to the drop down menu
// that offers the visualisation options
$(document).ready(function() {
  $('.dropbtn').click(function() {
    $(this).toggleClass('active');
    $(this).next('.dropdown-content').toggleClass('show');
  });

const audio = new Audio("https://cdn.pixabay.com/audio/2022/10/30/audio_4fa75b6720.mp3");
const buttons = document.querySelectorAll("button");
buttons.forEach(button => {
  button.addEventListener("click", () => {
    audio.play();
  });
});

const drop_buttons = document.querySelectorAll(".dropbtn");
drop_buttons.forEach(button => {
  button.addEventListener("click", () => {
    audio.play();
  });
});

  $(document).click(function(e) {
    var target = e.target;
    if (!$(target).is('.dropbtn') && !$(target).parents().is('.dropbtn')) {
      $('.dropbtn').removeClass('active');
      $('.dropdown-content').removeClass('show');
    }
  });
});
