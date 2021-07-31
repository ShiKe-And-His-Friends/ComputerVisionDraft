#include "webrtc/modules/audio_processing/aec/include/echo_cancellation.h"

int16_t *samples = (int16_t *)buffer->push();
if(samples == NULL){
	log_debug("drop oldest recorded frame");
	buffer->pop();
	samples = (int16_t *)buffer->push();
	assert(samples);
}
