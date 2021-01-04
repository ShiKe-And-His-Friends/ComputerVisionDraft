#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "raw/stb_image.h"
#include "raw/shader_s_1.h"
#include <iostream>


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
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	} else {
		std::cout << "Success to create GLFW window" << std::endl;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window ,framebuffer_size_callback);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	} else {
		std::cout << "Success initialize GLAD" << std::endl;	
	}

	Shader ourShader("./raw/4.1.texture.vs" ,"./raw/4.1.texture.fs");

	float vertices[] = {
		// position			// color			// texture coords
		0.5f ,0.5f ,0.0f 	,1.0f ,0.0f ,0.0f 	,1.0f ,1.0f
		,0.5f ,-0.5f ,0.0f 	,0.0f ,1.0f ,0.0f 	,1.0f ,0.0f
		,-0.5f ,-0.5f ,0.0f ,0.0f ,0.0f ,1.0f 	,0.0f ,0.0f
		,-0.5f ,0.5f ,0.0f	,1.0f ,1.0f ,0.0f	,0.0f ,1.0f
	};
	
	unsigned int indices[] = {
		0 ,1 ,3
		,1 ,2 ,3
	}

	unsigned int VBO ,VAO ,EBO;
	glGenVertexArrays(1 ,&VAO);
	glGenBuffers(1 ,&VBO);
	glGenBuffers(1 ,&EBO);
	
	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER ,VBO);
	glBufferData(GL_ARRAY_BUFFER ,sizeof(vertices) ,vertices ,GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER ,EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER ,sizeof(indices) ,indices ,GL_STATIC_DRAW);

	glVertexAttribPointer(0 ,3 ,GL_FLOAT ,GL_FALSE ,8 * sizeof(float) ,(void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1 ,3 ,GL_FLOAT ,GL_FALSE ,8 * sizeof(float) ,(void*)(3*sizeof(float)));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(2 ,2 ,GL_FLOAT ,GL_FALSE ,8 * sizeof(float) ,(void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);

	// load and create a texture
	unsigned int texture;
	glGenTextures(1 ,&texture);
	glBindTexture(GL_TEXTURE_2D ,texture);
	
	glTexParamteri(GL_TEXTURE_2D ,GL_TEXTURE_WARP_S ,GL_REPEAT);
	glTexParamteri(GL_TEXTURE_2D ,GL_TEXTURE_WARP_T ,GL_REPEAT);
	
	glTexParamteri(GL_TEXTURE_2D ,GL_TEXTURE_MIN_FILTER ,GL_LINEAR);
	glTexParamteri(GL_TEXTURE_2D ,GL_TEXTURE_MAX_FILTER ,GL_LINEAR);
	
	int width ,height ,nrChannels;
	unsigned char* data = stbi_load(FileSystem::getPath("drawable/container.jpg") ,&width ,&height ,&nrChannels ,0);
	if (data) {
		glImage2D(GL_TEXTURE_2D ,0 ,GL_RGB ,width ,height ,0 ,GL_RGB ,GL_UNSIGNED_BYTE ,data);
		glGenerateMipmap(GL_TEXTURE_2D);
	} else {
		std::out << "Failed to load texture" << std::endl;
	}
	stbi_image_free(data);
	
	while (!glfwWindowShouldClose(window)) {
		processInput(window);
		glClearColor(0.2f ,0.3f ,0.3f ,1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		
		ourShader.use();

		glBindVertexArray(VAO);
		glDrawElements(GL_TRIANGLES ,6 ,GL_UNSIGNED_INT,3);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glDeleteVertexArrays(1 ,&VAO);
	glDeleteBuffers(1 ,&VBO);
	glDeleteBuffers(1 ,&EBO);

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
