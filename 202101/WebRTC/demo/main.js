function hasUserMedia() {
	return !!(navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia);
}
if (hasUserMedia()) {
	navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;   
	navigator.getUserMedia({
		video: true,
		audio: true
	}, function (stream) {
		var video = document.querySelector('video');
		video.src = window.URL.createObjectURL(stream);
	}, function (err) {});
} else {
	alert("Expoler not support getUserMedia.");
}
navigator.getUserMedia({video:false ,audio:true } ,function(stream) {

});
