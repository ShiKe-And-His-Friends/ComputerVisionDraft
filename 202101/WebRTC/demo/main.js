function hasUserMedia() {
	return !!(navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia);
}
if (hasUserMedia()) {
	navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;   

navigator.getUserMedia(
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

}

var localUserId = Math.random().toString(36).subStr(2);
var room = prompt('Enter room name');
var socket = io('http://localhost:8080/socket.io');
if (room != '') {
	console.log('Attempted to join room:' ,localUserId ,room);
	var args = {
		'userId' : localUserId,
		'roomName' : room
	}
	socket.emit('join-room' ,JSON.stringfy(args))
}

socket.on('connect' ,function(){
	console.log("Signal server connected!")
});
socket.on('user-joined' ,function(userId) {
	if (localUserId == userId) {
		return;
	}
	console.log('Peer joined room:' ,userId);
});
