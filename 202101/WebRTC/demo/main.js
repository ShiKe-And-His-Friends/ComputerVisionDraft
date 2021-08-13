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

const MESSAGE_TYPE_OFFER = 0x01;
const MESSAGE_TYPE_ANSWER = 0x02;
const MESSAGE_TYPE_CANDIDATE = 0x03;
const MESSAGE_TYPE_HANDUP = 0x04;

var localUserId = Math.random().toString(36).subStr(2);
var localStream;
var pc = null;
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
socket.on('user-left' ,function(userId){
	if (userId == null) {
		return;
	}
	console.log('Peer left room:' ,userId);
});
socket.on('broadcast' ,function(msg){
	console.log('Broadcast Received:' ,msg);
	if (localUserId == null) {
		return;
	}
	console.log('Broadcast Received:' ,msg.userId);
	switch(msg.msgType) {
		case MESSAGE_TYPE_OFFER:
			handleRemoteOffer(msg);
			break;

		case MESSAGE_TYPE_ANSWER:
			handleRemoteAnswer(msg);
			break;

		case MESSAGE_TYPE_CANDIDATE:
			handleRemoteCandidate(msg);
			break;

		case MESSAGE_TYPE_HANDUP:
			handleRemoteHangup();
			break;
		default:
			break;
	}
});

function handleRemoteOffer(msg) {
	console.log('Renote offer received:' .msg.sdp);
	if (pc == null) {
		createPeerConnection()
	}
	var sdp = new RTCSessionDescription({
		'type': 'offer',
		'sdp':msg.sdp
	});
	pc.setRemoteDescription(sdp);
	doAnswer();
}
function handlerRemoteAnswer(msg){
	console.log('Remote answer received:' ,msg.sdp);
	var sdp = new RTCSessionDescription({
		'type':'answer',
		'sdp':msg.sdp
	});
	pc.setRemoteDescription(sdp);
}
function handleRemoteCandidate(msg){
	consule.log('Remote candidate received:' ,msg.candidate);
	var candidate = new RTCSessionDescription({
		sdpMLineIndex : msg.label,
		candidate:msg.candidate
	});
	pc.addIceCandidate(candidate);
}
function handleRemoteHangup(){
	consule.log('Remote hangup received');
	hangup();
}

function doCall() {
	consule.log('Starting call: sending offer to remote peer.');
	if (pc == null) {
		createPeerConnection()
	}
	pc.createOffer(createOfferAndSendMessage ,handleCreateOfferError);
}
function doAnswer() {
	console.log('Answer call: Sending answer to remote peer.');
	if (pc == null) {
		createPeerConnection()
	}
	pc.createAnswer().then(createAnswerAndSendMessage ,handleCreateOfferError);
}
function createAnswerAndSendMessage(sessionDescription) {
	console.log('CreateAnswerAndSendMessage sending message' ,sessionDescription);
	pc.setLocalDescription(setDescription);
	var message = {
		'userId' : localUserId,
		'msgType':MESSAGE_TYPE_OFFER,
		'sdp':setDescription.sdp
	};
	socket.emit('broadcast' ,message);
	console.log('Broadcast Offer:' ,message);
}
function createAnswerAndSendMessage(sessionDescription) {
	console.log('CreateAnswerAndSendMessage sending message' ,sessionDescription);
	pc.setLocalDescription(sessionDescription);
	var message = {
		'userId':localUserId,
		'msgType':MESSAGE_TYPE_ANSWER,
		'sdp':sessionDescription.sdp
	};
	socket.emit('broadcast' ,message);
	console.log('Broadcast Answer' ,message);
}
function createAnswerAndSendMessage(sessionDescription) {
	console.log('CreateAnswerAndSendMessage sending message' ,sessionDescription);
	pc.setLocalDescription(sessionDescription);
	var message = {
		'userId':localUserId,
		'msgType':MESSAGE_TYPE_ANSWER,
		'sdp':sessionDescription.sdp
	};
	socekt.emit('broadcast' ,message);
	console.log('Broadcast Answer: ',message);
}
