var toggleStatus = 1;
function toggleMenu() {
  if (toggleStatus == 1 ) { 
    document.getElementById("mySidebar").style.display = "block";
    document.getElementById("overlay").style.display = "block";
    toggleStatus = 0;
  }
      
  else if(toggleStatus == 0) {
    document.getElementById("mySidebar").style.display = "none";
    document.getElementById("overlay").style.display = "none";
    toggleStatus = 1;
    }
}