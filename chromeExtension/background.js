var result;
$.ajax({
  url: "glove_small_dict.json",
}).done(function(results) {
  result = $.parseJSON(results)
  console.log(result.padua);
  chrome.runtime.onMessage.addListener( function(request, sender, sendResponse) { 
	
	console.log(request.arrays);
	var dataSet = Object.keys(result);
	var vectorObject = {};
	request.arrays.forEach(function(word) {
		if(dataSet.indexOf(word)!=-1){
			vectorObject[word] = result[word];
			// console.log(word+"....."+result[word]);
		}
	});console.log(vectorObject);
	
	sendResponse({vectorObject});

	});
});
