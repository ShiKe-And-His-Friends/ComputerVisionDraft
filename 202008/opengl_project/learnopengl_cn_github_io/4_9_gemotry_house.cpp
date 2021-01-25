#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#include "raw/stb_image.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "raw/shader_s_2.h"
#include <iostream>

void framebuffer_size_callback(GLFWwindow* window ,int width ,int height);

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

Camera camera(glm::vec3(0.0f ,0.0f ,3.0f));
float lastX = SCR_WIDTH /2.0f;
float lastY = SCR_HEIGHT / 2.0f;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

int main () {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR ,3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR ,3);
	glfwWindowHint(GLFW_OPENGL_PROFILE ,GLFW_OPENGL_CORE_PROFILE);
	
#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT ,GL_TRUE);
#endif

	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH ,SCR_HEIGHT ,"LearnOpenGL" ,NULL ,NULL);
	if (window == NULL) {
		std::cout << "GLFW Window init failure." << std::endl;
		glfwTerminate();
		return -1;
	} else {
		std::cout << "GLFW Window init success." << std::endl;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window ,framebuffer_size_callback);
	glfwSetCursorPosCallback(window ,mouse_callback);
	
	glfwSetInputMode(window ,GLFW_CURSOR ,GLFW_CURSOR_DISABLED);
	
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cout << "Initialize GLAD failure." << std::endl;
		return -1;
	} else {
		std::cout << "Initialize GLAD success." << std::endl;
	}
	
	glEnable(GL_DEPTH_TEST);
	
	Shader shader("./raw/9.1.geometry_shader.vs" ,"./raw/9.1.geometry_shader.fs" ,"./raw/9.1.geometry_shader.gs");

	float points[] = {
		-0.5f,  0.5f, 1.0f, 0.0f, 0.0f, // top-left
		0.5f,  0.5f, 0.0f, 1.0f, 0.0f, // top-right
		0.5f, -0.5f, 0.0f, 0.0f, 1.0f, // bottom-right
		-0.5f, -0.5f, 1.0f, 1.0f, 0.0f  // bottom-left
	};
	
	unsigned int VBO ,VAO;
	glGenBuffers(1 ,&VBO);
	glGenVertexArrays(1 ,&VAO);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER ,VBO);
	glBufferData(GL_ARRAY_BUFFER ,sizeof(points) ,&points ,GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertrixPointer(0 ,2 ,GL_FLOAT ,GL_FLOAT ,5 * sizeof(float) ,0);
	glEnableVertexAttribArray(1);
	glVertrixPointer(1 ,3 ,GL_FLOAT ,GL_FLOAT ,5 * sizeof(float) ,(void*)(2 * sizeof(float)));
	glBindVertexArray(0);
	
	
	while(!glfwWindowShouldClose(window)) {
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		shader.use();
		glBindVertexArray(VAO);
		glDrawArrays(GL_POINTS ,0 ,4);
		
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	
	glDeleteVertexArrays(1 ,&cubeVAO);
	glDeleteBuffers(1 ,&cubeVBO);
	
	glfwTerminate();
	return 0;
}

void framebuffer_size_callback(GLFWwindow* window ,int width ,int height) {
	glViewport(0 ,0 ,width ,height);
}