#include <glad/glad.h>
#include <GLFW/glfw3.h>
#define STB_IMAGE_IMPLEMENTATION
#include "raw/stb_image.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "raw/shader_s_1.h"
#include <iostream>
#include <filesystem>

/**
 sudo apt-get install libglm-dev
 */

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

	Shader ourShader("./raw/5.2.texture.vs" ,"./raw/5.2.texture.fs");

	float vertices[] = {
		// position			// texture coords			
		0.5f ,0.5f ,0.0f 	,1.0f ,1.0f
		,0.5f ,-0.5f ,0.0f 	,1.0f ,0.0f
		,-0.5f ,-0.5f ,0.0f ,0.0f ,0.0f
		,-0.5f ,0.5f ,0.0f	,0.0f ,1.0f
	};
	
	unsigned int indices[] = {
		0 ,1 ,3
		,1 ,2 ,3
	};

	unsigned int VBO ,VAO ,EBO;
	glGenVertexArrays(1 ,&VAO);
	glGenBuffers(1 ,&VBO);
	glGenBuffers(1 ,&EBO);
	
	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER ,VBO);
	glBufferData(GL_ARRAY_BUFFER ,sizeof(vertices) ,vertices ,GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER ,EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER ,sizeof(indices) ,indices ,GL_STATIC_DRAW);

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
		glTexImage2D(GL_TEXTURE_2D ,0 ,GL_RGBA ,width ,height ,0 ,GL_RGBA ,GL_UNSIGNED_BYTE ,data);
		glGenerateMipmap(GL_TEXTURE_2D);
	} else {
		std::cout << "Failed to load texture" << std::endl;
	}
	stbi_image_free(data);
	ourShader.use();
	glUniform1i(glGetUniformLocation(ourShader.ID ,"texture1") ,0);
	ourShader.setInt("texture2" ,1);
	
	while (!glfwWindowShouldClose(window)) {
		processInput(window);
		glClearColor(0.2f ,0.3f ,0.3f ,1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D ,texture);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D ,texture2);
		

		glm::mat4 transform = glm::mat4(1.0f);

		transform = glm::translate(transform ,glm::vec3(0.5f ,-0.5f ,0.0f));
		transform = glm::rotate(transform ,(float)glfwGetTime() ,glm::vec3(0.0f ,0.0f ,1.0f));
		unsigned int transformLoc = glGetUniformLocation(ourShader.ID ,"transform");
		glUniformMatrix4fv(transformLoc ,1 ,GL_FALSE ,glm::value_ptr(transform));
		
		glBindVertexArray(VAO);
		glDrawElements(GL_TRIANGLES ,6 ,GL_UNSIGNED_INT,0);

		// second transformation
		transform = glm::mat4(1.0f);
		transform = glm::translate(transform ,glm::vec3(-0.5 ,0.5 ,0.0f));
		float scaleAmount = sin(glfwGetTime());
		transform = glm::scale(transform ,glm::vec3(scaleAmount ,scaleAmount ,scaleAmount));
		glUniformMatrix4fv(transformLoc ,1 ,GL_FALSE ,&transform[0][0]);

		glDrawElements(GL_TRIANGLES ,6 ,GL_UNSIGNED_INT ,0);

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
