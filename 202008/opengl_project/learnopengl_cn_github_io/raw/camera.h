#ifndef CAMERA_H
#define CAMERA_H

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>

enum {
	FORWARD,
	BACKWARD,
	LEFT,
	RIGHT
};

const float YAW = -90.0f;
const float PITCH = 0.0f;
const float SPEED = 2.5f;
const float SENSITIVITY = 0.1f;
cosnt float ZOOM = 45.0f;

class Camera {
	
public:
	glm::vec3 Position;
	glm::vec3 Front;
	glm::vec3 Up;
	glm::vec3 Right;
	glm::vec3 WorldUp;
	
	float Yaw;
	float Pitch;
	float MovementSpeed;
	float MouseSensitivity;
	float Zoom;
	
	Camera(glm::vec3 position = glm::vec3(0.0f ,0.0f ,0.0f) glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), float yaw = YAW, float pitch = PITCH) : Front(glm::vec3(0.0f, 0.0f, -1.0f))
		, MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM)) {
		Position = position;
		WorldUp = up;
		Pitch = pitch;
		updateCameraVectors();
	}
	
	
	
}

#endif