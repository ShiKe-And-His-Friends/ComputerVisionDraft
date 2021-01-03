#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "shader_s_1.h"
#include <iostream>

using namespace std;

void framebuffer_size_callback(GLFWwindow* window ,int width ,int height);
void processInput(GLFWwindow* windows);

const unsigned int SRC_WIDTH = 800;
const unsigned int SRC_HEIGHT = 600;

int main () {

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR ,3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR ,3);
	glfwWindowHint(GLFW_OPENGL_PROFILE ,GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT ,GL_TRUE);
	cout << "Initialize Forward apple compat." << endl;
#endif

	GLFWwindow* window = glfwCreateWindow(800 ,600 ,"LearnOpenGl" ,NULL ,NULL);
	if (window == NULL) {
		cout << "Failed to create GLFW window" << endl;
		glfwTerminate();
		return -1;
	} else {
		cout << "Success to create GLFW window" << endl;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window ,framebuffer_size_callback);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		cout << "Failed to initialize GLAD" << endl;
		return -1;
	} else {
		cout << "Success initialize GLAD" << endl;	
	}

	Shader outShader("3.3.shader.vs" ,"3.3.shader.fs");

	float vertices[] = {
		// position	// color
		0.5f ,-0.5f ,0.0f ,1.0f ,0.0f ,0.0f
		,-0.5f ,-0.5f ,0.0f ,0.0f ,1.0f ,0.0f
		,0.0f ,0.5f ,0.0f ,0.0f ,0.0f ,1.0f
	};

	unsigned int VBO ,VAO;
	glGenVertexArrays(1 ,&VAO);
	glGenBuffers(1 ,&VBO);
	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER ,VBO);
	glBufferData(GL_ARRAY_BUFFER ,sizeof(vertices) ,vertices ,GL_STATIC_DRAW);

	glVertexAttribPointer(0 ,3 ,GL_FLOAT ,GL_FALSE ,6 * sizeof(float) ,(void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1 ,3 ,GL_FLOAT ,GL_FALSE ,6 * sizeof(float) ,(void*)(3*sizeof(float)));
	glEnableVertexAttribArray(1);

	glUseProgram(shaderProgram);

	while (!glfwWindowShouldClose(window)) {
		processInput(window);
		glClearColor(0.2f ,0.3f ,0.3f ,1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		outShader.use();

		glBindVertexArray(VAO);
		glDrawArrays(GL_TRIANGLES ,0 ,3);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glDeleteVertexArrays(1 ,&VAO);
	glDeleteBuffers(1 ,&VBO);
	glDeleteProgram(shaderProgram);

	glfwTerminate();
	return 0;
}

void processInput(GLFWwindow* window) {
	if (glfwGetKey(window ,GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window ,true);
	}
}

void framebuffer_size_callback(GLFWwindow* window ,int width ,int height) {
	glViewport(0 ,0 ,width ,height);
}
