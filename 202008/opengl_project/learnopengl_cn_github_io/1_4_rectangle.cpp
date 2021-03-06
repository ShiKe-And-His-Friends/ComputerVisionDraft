#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

using namespace std;

void framebuffer_size_callback(GLFWwindow* window ,int width ,int height);
void processInput(GLFWwindow* windows);

const unsigned int SRC_WIDTH = 800;
const unsigned int SRC_HEIGHT = 600;

const char* vertexShaderSource = "#version 330 core\n"
	"layout (location = 0) in vec3 aPos;\n"
	"void main() \n"
	"{\n"
	"  gl_Position = vec4(aPos.x ,aPos.y ,aPos.z ,1.0);\n"
	"}\0";

const char* fragmentShaderSource = "#version 330 core\n"
	"out vec4 FragColor;\n"
	"void main()\n"
	"{\n"
	"  FragColor = vec4(1.0f ,0.5f ,0.2f ,1.0f);\n"
	"}\n\0";

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

	int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader ,1 ,&vertexShaderSource ,NULL);
	glCompileShader(vertexShader);
	int success;
	char infoLog[512];
	glGetShaderiv(vertexShader ,GL_COMPILE_STATUS ,&success);
	if (!success) {
		glGetShaderInfoLog(vertexShader ,512 ,NULL ,infoLog);
		cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << endl;
	}
	int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader ,1 ,&fragmentShaderSource ,NULL);
	glCompileShader(fragmentShader);
	glGetShaderiv(fragmentShader ,GL_COMPILE_STATUS ,&success);
	if (!success) {
		glGetShaderInfoLog(fragmentShader ,512 ,NULL ,infoLog);
		cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << endl;
	}
	int shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram ,vertexShader);
	glAttachShader(shaderProgram ,fragmentShader);
	glLinkProgram(shaderProgram);
	glGetProgramiv(shaderProgram ,GL_LINK_STATUS ,&success);
	if (!success) {
		glGetProgramInfoLog(shaderProgram ,512 ,NULL ,infoLog);
		cout << "ERROR::SHADER::PROGRAME::COMPILATION_FAILED\n" << infoLog << endl;
	}
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
	
	float vertices[] = {
		0.5f ,0.5f ,0.0f
		,0.5f ,-0.5f ,0.0f
		,-0.5f ,-0.5f ,0.0f
		,-0.5f ,0.5f ,0.0f
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


	glVertexAttribPointer(0 ,3 ,GL_FLOAT ,GL_FALSE ,3 * sizeof(float) ,(void*)0);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER ,0);

	glBindVertexArray(0);

	glPolygonMode(GL_FRONT_AND_BACK ,GL_LINE);
	while (!glfwWindowShouldClose(window)) {
		processInput(window);
		glClearColor(0.2f ,0.3f ,0.3f ,1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		glUseProgram(shaderProgram);
		glBindVertexArray(VAO);
		// glDrawArrays(GL_TRIANGLES ,0 ,6);
		glDrawElements(GL_TRIANGLES ,6 ,GL_UNSIGNED_INT ,0);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glDeleteVertexArrays(1 ,&VAO);
	glDeleteBuffers(1 ,&VBO);
	glDeleteBuffers(1 ,&EBO);
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
