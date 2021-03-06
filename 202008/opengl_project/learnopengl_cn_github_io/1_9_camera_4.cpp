#include <glad/glad.h>
#include <GLFW/glfw3.h>
#define STB_IMAGE_IMPLEMENTATION
#include "raw/stb_image.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "raw/shader_s_2.h"
#include "raw/camera_2.h"
#include <iostream>
#include <filesystem>

/**
 sudo apt-get install libglm-dev
 */

void framebuffer_size_callback(GLFWwindow* window ,int width ,int height);
void mouse_callback(GLFWwindow* window ,double xpos ,double ypos);
void scroll_callback(GLFWwindow* window ,double xoffset ,double yoffset);
void processInput(GLFWwindow* windows);

const unsigned int SRC_WIDTH = 800;
const unsigned int SRC_HEIGHT = 600;

Camera camera(glm::vec3(0.0f ,0.0f ,3.0f));
bool firstMouse = true;
float lastX = SRC_WIDTH / 2.0f;
float lastY = SRC_HEIGHT / 2.0f;

float deletaTime = 0.0f;
float lastFrame = 0.0f;

int main () {

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR ,3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR ,3);
	glfwWindowHint(GLFW_OPENGL_PROFILE ,GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT ,GL_TRUE);
	cout << "Initialize Forward apple compat." << endl;
#endif

	GLFWwindow* window = glfwCreateWindow(SRC_WIDTH ,SRC_HEIGHT ,"LearnOpenGl" ,NULL ,NULL);
	if (window == NULL) {
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	} else {
		std::cout << "Success to create GLFW window" << std::endl;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetScrollCallback(window ,scroll_callback);
	
	glfwSetInputMode(window ,GLFW_CURSOR ,GLFW_CURSOR_DISABLED);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	} else {
		std::cout << "Success initialize GLAD" << std::endl;	
	}

	glEnable(GL_DEPTH_TEST);

	Shader ourShader("./raw/7.4.camera.vs" ,"./raw/7.4.camera.fs");

	float vertices[] = {
		// position			// texture coords			
		-0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,

        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,

        -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  1.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,

        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f
	};
	
	glm::vec3 cubePositions[] = {
        glm::vec3( 0.0f,  0.0f,  0.0f),
        glm::vec3( 2.0f,  5.0f, -15.0f),
        glm::vec3(-1.5f, -2.2f, -2.5f),
        glm::vec3(-3.8f, -2.0f, -12.3f),
        glm::vec3 (2.4f, -0.4f, -3.5f),
        glm::vec3(-1.7f,  3.0f, -7.5f),
        glm::vec3( 1.3f, -2.0f, -2.5f),
        glm::vec3( 1.5f,  2.0f, -2.5f),
        glm::vec3( 1.5f,  0.2f, -1.5f),
        glm::vec3(-1.3f,  1.0f, -1.5f)
	};

	unsigned int VBO ,VAO;
	glGenVertexArrays(1 ,&VAO);
	glGenBuffers(1 ,&VBO);
	
	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER ,VBO);
	glBufferData(GL_ARRAY_BUFFER ,sizeof(vertices) ,vertices ,GL_STATIC_DRAW);

	glVertexAttribPointer(0 ,3 ,GL_FLOAT ,GL_FALSE ,5 * sizeof(float) ,(void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1 ,2 ,GL_FLOAT ,GL_FALSE ,5 * sizeof(float) ,(void*)(3*sizeof(float)));
	glEnableVertexAttribArray(1);

	// load and create a texture
	unsigned int texture ,texture2;
	glGenTextures(1 ,&texture);
	glBindTexture(GL_TEXTURE_2D ,texture);
	
	glTexParameteri(GL_TEXTURE_2D ,GL_TEXTURE_WRAP_S ,GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D ,GL_TEXTURE_WRAP_T ,GL_REPEAT);
	
	glTexParameteri(GL_TEXTURE_2D ,GL_TEXTURE_MIN_FILTER ,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D ,GL_TEXTURE_MAG_FILTER ,GL_LINEAR);
	
	int width ,height ,nrChannels;
	stbi_set_flip_vertically_on_load(true);
	unsigned char* data = stbi_load("/home/shike/Documents/computerVisionDraft/202008/opengl_project/learnopengl_cn_github_io/drawable/container.jpg" ,&width ,&height ,&nrChannels ,0);
	if (data) {
		glTexImage2D(GL_TEXTURE_2D ,0 ,GL_RGB ,width ,height ,0 ,GL_RGB ,GL_UNSIGNED_BYTE ,data);
		glGenerateMipmap(GL_TEXTURE_2D);
	} else {
		std::cout << "Failed to load texture" << std::endl;
	}
	stbi_image_free(data);

	// texture 2
	glGenTextures(1 ,&texture2);
	glBindTexture(GL_TEXTURE_2D ,texture2);
	glTexParameteri(GL_TEXTURE_2D ,GL_TEXTURE_WRAP_S ,GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D ,GL_TEXTURE_WRAP_T ,GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D ,GL_TEXTURE_MIN_FILTER ,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D ,GL_TEXTURE_MAG_FILTER ,GL_LINEAR);

	data = stbi_load("/home/shike/Documents/computerVisionDraft/202008/opengl_project/learnopengl_cn_github_io/drawable/awesomeface.png" ,&width ,&height ,&nrChannels ,0);
	if (data) {
		glTexImage2D(GL_TEXTURE_2D ,0 ,GL_RGB ,width ,height ,0 ,GL_RGBA ,GL_UNSIGNED_BYTE ,data);
		glGenerateMipmap(GL_TEXTURE_2D);
	} else {
		std::cout << "Failed to load texture" << std::endl;
	}
	stbi_image_free(data);
	ourShader.use();
	ourShader.setInt("texture1" ,0);
	ourShader.setInt("texture2" ,1);
	
	while (!glfwWindowShouldClose(window)) {
		float currentFrame = glfwGetTime();
		deletaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;
		
		processInput(window);
		glClearColor(0.2f ,0.3f ,0.3f ,1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D ,texture);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D ,texture2);
		
		ourShader.use();

		glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom) ,(float)SRC_WIDTH / (float)SRC_HEIGHT ,0.1f ,100.0f);
		ourShader.setMat4("projection" ,projection);
		
		glm::mat4 view = camera.GetViewMatrix();
		ourShader.setMat4("view" ,view);
	
		glBindVertexArray(VAO);
		for (unsigned int i = 0 ; i < 10 ;i++) {
			glm::mat4 model = glm::mat4(1.0f);
			model = glm::translate(model ,cubePositions[i]);
			float angle = 20.0f * i;
			model = glm::rotate(model ,glm::radians(angle) ,glm::vec3(1.0f ,0.3f ,0.5f));
			ourShader.setMat4("model" ,model);
			glDrawArrays(GL_TRIANGLES ,0 ,36);
		}

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glDeleteVertexArrays(1 ,&VAO);
	glDeleteBuffers(1 ,&VBO);

	glfwTerminate();
	return 0;
}

void processInput(GLFWwindow* window) {
	if (glfwGetKey(window ,GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window ,true);
	}
	float cameraSpeed = 2.5 * deletaTime;
	if (glfwGetKey(window ,GLFW_KEY_W) == GLFW_PRESS) {
		camera.ProcessKeyBoard(FORWARD ,deletaTime);
	}
	if (glfwGetKey(window ,GLFW_KEY_S) == GLFW_PRESS) {
		camera.ProcessKeyBoard(BACKWARD ,deletaTime);
	}
	if (glfwGetKey(window ,GLFW_KEY_A) == GLFW_PRESS) {
		camera.ProcessKeyBoard(LEFT ,deletaTime);
	}
	if (glfwGetKey(window ,GLFW_KEY_D) == GLFW_PRESS) {
		camera.ProcessKeyBoard(RIGHT ,deletaTime);
	}
}

void framebuffer_size_callback(GLFWwindow* window ,int width ,int height) {
	glViewport(0 ,0 ,width ,height);
}

void mouse_callback(GLFWwindow* window ,double xpos ,double ypos) {
	if (firstMouse) {
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}
	
	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos;
	lastX = xpos;
	lastY = ypos;
	
	camera.ProcessMouseMovement(xoffset ,yoffset);
}

void scroll_callback(GLFWwindow* window ,double xoffset ,double yoffset) {
	camera.ProcessMouseScroll(yoffset);
}
