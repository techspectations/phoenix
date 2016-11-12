// var vectors;
// $.getJSON("./glove_small_dict.json", function(json) {
// vectors = json;
// });
// console.log(vectors);

// $.ajax({
//   url: "glove_small_dict.json",
// }).done(function(results) {
//   result = $.parseJSON(results)

$(document).ready(function () {
  $("#searchinput").focus();

  $("#searchinput").keyup(function(event) {
      event.preventDefault();
      var keycode = (event.keyCode ? event.keyCode : event.which);
      if (keycode == '13') {
          if ($("#searchinput").val() != "") {
              $("#search-error-msg").css("display", "none");
              var searchWord = $("#searchinput").val();
              console.log(searchWord);
              
              chrome.tabs.query({
                  currentWindow: true,
                  active: true
              }, function(tabs) {
                  var activeTab = tabs[0];
                  chrome.tabs.sendMessage(activeTab.id, {
                      "message": searchWord
                  });
              });
          }else {

          $("#search-error-msg").css("display", "block");
          }
      } 

  });

});
  