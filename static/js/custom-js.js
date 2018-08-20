function myFunction() {
    if (document.getElementById("demo").style.color == "blue") {
        document.getElementById("demo").style.color = "red";
    } else {
        document.getElementById("demo").style.color = "blue"
    }    
}

function sendPost(){
    if (document.getElementById("demo").style.color == "blue") {
        document.getElementById("demo").style.color = "red";
    } else {
        document.getElementById("demo").style.color = "blue"
    }  
    var xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) 
        {
             //do what you will with the response
        }
    };
    xhttp.open("POST", "/user_input", true);
    xhttp.send("fuck");
}