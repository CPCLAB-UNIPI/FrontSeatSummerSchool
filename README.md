The following steps will get you started for the exercise sessions:

1. Clone this repository (green button "Code")
2. Install VSCode: https://code.visualstudio.com/
3. Install Docker Desktop for your operating system:
   [Ubuntu](https://docs.docker.com/desktop/install/ubuntu/), [Other Linux](https://docs.docker.com/desktop/install/linux-install/), [Windows](https://docs.docker.com/desktop/install/windows-install/), [MacOS](https://docs.docker.com/desktop/install/mac-install/).
   On Linux, you may need to follow the [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/) steps for
   plotting.
4. Install the app Dev Containers in VSCode:
        Open the Extension Marketplace: https://code.visualstudio.com/docs/editor/extension-marketplace
        Search and install Dev Containers
5. Start VSCode with the top folder of this repository as the root folder:
        A message should appear on the bottom right corner asking you to build the container.
        Alternatively, you can open the Command Palette of VSCode (on the left bottom corner, under the Manage button - "dented wheel" icon) and type in “Dev Containers: Rebuild and Reopen in Container.”.
        Click on it to start building the image and wait until the container runs. Depending on your computer and internet connection, the process can take several minutes.
        You can check that you are running the dev container by looking at the bottom left corner where a field should state “Dev Container: DevContainerFSSS”.

6. Test the container: Open an example: `casadi_test.py` or `mpc_acados/minimal_example_ocp.py`
       Open the command palette (Crtl+Shift+p) -> "Python: Run Python file in terminal" -> enter

7. ML Workshop day 3. - Neuromancer package
A Google account is required to access the Google Colab. If you want to save the changes, click on "Copy to Drive" button at the top of the page. You can run each command individually, by pressing the play button at each line (Ctrl+Enter) or you can run the whole code by clicking on "Runtime" -> "Run all"

