function hasUserMedia() {
	return !!(navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia);
}
if (hasUserMedia()) {
	navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;   

navigator.mediaDevices.getUserMedia(
	{
		video:{
			mandatory: {
				minAspectRation: 1.777,
				maxAspectRation: 1.778
			},
			optional: {
				maxWidth: 640,
				maxHeigth: 480
			}
		} ,
		audio:false
	} ,function(stream) {
		var video = document.querySelector('video');
		video.src = window.URL.createObjectURL(stream);
	} ,function(error) {
		console.log("Raised an error when capturing" ,error);
	}
);
	
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
