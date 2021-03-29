# Image Understanding
## Curso 2020-21
La practica consiste en crear un clasificador que identifique el cancer en imágenes de histologías colorectales.


## Instalacion Local

Primero, hay que instalar python. Se puede hacer desde [aquí](https://www.python.org/downloads/). En sistemas basados en Debian también se puede hacer mediante la terminal con
`sudo apt-get install python3.8`.

Luego hay que instalar `pip` si no se ha instalado junto con python.

Además, para evitar tener problemas de dependencias es mejor crear un entorno virtual ligero. Para ello, instalaremos `virtualenv`. Se puede hacer desde cualquier terminal mediante le comando `pip install virtualenv`. 

En la carpeta del repositorio, se deberá hacer lo siguiente:

```
virtualenv <DIR> # Esto creará el entorno virtual
```
Si estás usando una máquina UNIX:
`source <DIR>/bin/activate`

Si es una máquina Win32: 

```
cd <DIR>\Scripts\
.\activate
```

Se debería observar que el prompt cambia. Ahora se deberá instalar `scipy` mediante el comando `pip install scipy`, y finalmente `ipykernel` mediante `pip install ipykernel` para poder crear Cuadernos de Jupyter.

Recomiendo utilizar [Visual Studio Code](https://code.visualstudio.com/) con la extensión oficial de Python (ms-python.python), ya que con eso podéis utilizar todo. Para configurarlo, debéis ir a la paleta de comandos (por defecto, Ctrl + Shift + P) y abrir las opciones en modo de interfaz ("Preferences: Open Settings (UI)").

Una vez ahí, podéis buscar "virtual" y os aparecerán dos opciones. Si esto no es así, podéis ir a la seccion Extensiones->Python. De una forma u otra, debéis encontrar la subsección que pone "Python: Venv Folders" y añadir el path del repositorio. Ahora debemos reiniciar Visual Studio Code. Esto último es necesario porque Virtual Env lo que hace es crear una nueva instalación de python prescindible sobre la que realizar todas las modificaciones que queramos, sin afectar a la instalación global. Por defecto, VSCode no detecta esta instalación, sino la global, y debemos decirle que tiene que elegir la otra.

Lo último que es necesario para que quede todo configurado es acceder de nuevo a la paleta de comandos, y ahí escribir "Python: Select Interpreter". Debería aparecer una opción. Una vez seleccionada, pulsamos ENTER y seleccionamos la versión de python que esté en el repositorio. Asi mismo, al abrir por primera vez cualquier archivo de Jupyter (con la extensión .ipynb) tendremos que seleccionar el interprete adecuado en dandole click a la esquina superior derecha, al lado de donde pondrá "Jupyter Server".


