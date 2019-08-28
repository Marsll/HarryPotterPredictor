function myFunction() {
    if (document.getElementById("demo").style.color == "blue") {
        document.getElementById("demo").style.color = "red";
    } else {
        document.getElementById("demo").style.color = "blue"
    }    
}
function sendPost()
{
    var req = new XMLHttpRequest()
    req.onreadystatechange = function()
    {
        if (req.readyState == 4)
        {
            if (req.status != 200)
            {
                //error handling code here
            }
            else
            {
                var response = JSON.parse(req.responseText)
                document.getElementById('myDiv').innerHTML = response.username
            }
        }
    }

    req.open('POST', '/user_input')
    req.setRequestHeader("Content-type", "application/x-www-form-urlencoded")
    var text = document.getElementById("myInput").value
    var postVars = 'input='+ text
    req.send(postVars)
}
/* function sendPost(){
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
} */