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

const char* fragmentShader1Source = "#version 330 core\n"
	"out vec4 FragColor;\n"
	"void main()\n"
	"{\n"
	"  FragColor = vec4(1.0f ,0.5f ,0.2f ,1.0f);\n"
	"}\n\0";
const char* fragmentShader2Source = "#version 330 core\n"
	"out vec4 FragColor;\n"
	"void main()\n"
	"{\n"
	"  FragColor = vec4(1.0f ,1.0f ,0.0f ,1.0f);\n"
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

	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	unsigned int fragmentShaderOrange = glCreateShader(GL_FRAGMENT_SHADER);
	unsigned int fragmentShaderYellow = glCreateShader(GL_FRAGMENT_SHADER);
	unsigned int shaderProgramOrange = glCreateProgram();
	unsigned int shaderProgramYellow = glCreateProgram();

	glShaderSource(vertexShader ,1 ,&vertexShaderSource ,NULL);
	glCompileShader(vertexShader);
	glShaderSource(fragmentShaderOrange ,1 ,&fragmentShader1Source ,NULL);
	glCompileShader(fragmentShaderOrange);
	glShaderSource(fragmentShaderYellow ,1 ,&fragmentShader2Source ,NULL);
	glCompileShader(fragmentShaderYellow);

	glAttachShader(shaderProgramOrange ,vertexShader);
	glAttachShader(shaderProgramOrange ,fragmentShaderOrange);
	glLinkProgram(shaderProgramOrange);

	glAttachShader(shaderProgramYellow ,vertexShader);
	glAttachShader(shaderProgramYellow ,fragmentShaderYellow);
	glLinkProgram(shaderProgramYellow);

	
	float firstTriangle[] = {
		-0.9f ,-0.5f ,0.0f
		,-0.0f ,-0.5f ,0.0f
		,-0.45f ,0.5f ,0.0f
	};
	float secondTriangle[] = {
		0.0f ,-0.5f ,0.0f
		,0.9f ,-0.5f ,0.0f
		,0.45f ,0.5f ,0.0f
	};

	unsigned int VBOs[2] ,VAOs[2];
	glGenVertexArrays(2 ,VAOs);
	glGenBuffers(2 ,VBOs);

	glBindVertexArray(VAOs[0]);
	glBindBuffer(GL_ARRAY_BUFFER ,VBOs[0]);
	glBufferData(GL_ARRAY_BUFFER ,sizeof(firstTriangle) ,firstTriangle ,GL_STATIC_DRAW);
	glVertexAttribPointer(0 ,3 ,GL_FLOAT ,GL_FALSE ,3 * sizeof(float) ,(void*)0);
	glEnableVertexAttribArray(0);

	glBindVertexArray(VAOs[1]);
	glBindBuffer(GL_ARRAY_BUFFER ,VBOs[1]);
	glBufferData(GL_ARRAY_BUFFER ,sizeof(secondTriangle) ,secondTriangle ,GL_STATIC_DRAW);
	glVertexAttribPointer(0 ,3 ,GL_FLOAT ,GL_FALSE ,0 ,(void*)0);
	glEnableVertexAttribArray(0);


	while (!glfwWindowShouldClose(window)) {
		processInput(window);
		glClearColor(0.2f ,0.3f ,0.3f ,1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		glUseProgram(shaderProgramOrange);
		glBindVertexArray(VAOs[0]);
		glDrawArrays(GL_TRIANGLES ,0 ,3);

		glUseProgram(shaderProgramYellow);
		glBindVertexArray(VAOs[1]);
		glDrawArrays(GL_TRIANGLES ,0 ,3);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glDeleteVertexArrays(2 ,VAOs);
	glDeleteBuffers(2 ,VBOs);
	glDeleteProgram(shaderProgramOrange);
	glDeleteProgram(shaderProgramYellow);

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
