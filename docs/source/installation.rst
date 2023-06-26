Installation
=============================

This guide will walk you through the installation process for Albumentations3D, a Python package for 3D image augmentation specifically designed for volumetric 3D images like CT scans.






Prerequisites
-------------------------

Before installing Albumentations3D, ensure you have the following prerequisites:

- Python 3.7 or higher

Installation Steps
-------------------------

Follow these steps to install Albumentations3D:

1. Open a terminal or command prompt.

2. Create a new virtual environment (optional but recommended):

    .. code-block:: bash

        python -m venv myenv


3. Activate the virtual environment:

   - On Windows:

        .. code-block:: powershell

                myenv\Scripts\activate

   - On macOS and Linux:

        .. code-block:: bash

                source myenv/bin/activate

4. Install Albumentations3D and its dependencies using `pip`:

    .. code-block:: bash

        pip install albumentations3d


This will install Albumentations3D along with the required dependencies.

5. Verify the installation by importing Albumentations3D in Python:

    .. code-block:: python

        import albumentations3d as A


If no errors occur, the installation was successful.

Upgrade to the Latest Version
-----------------------------------------

To upgrade to the latest version of Albumentations3D, use the `--upgrade` flag with `pip`:

    .. code-block:: bash
        
        pip install --upgrade albumentations3d


This will update Albumentations3D to the latest available version.

Uninstall Albumentations3D
----------------------------------------

To uninstall Albumentations3D, use the following `pip` command:

    .. code-block:: bash
        
        pip uninstall albumentations3d

Confirm the uninstallation when prompted.

Additional Notes
------------------------------

- It's recommended to install Albumentations3D in a virtual environment to isolate it from other Python packages and prevent conflicts.

- If you encounter any issues during installation, please seek help from the Albumentations3D community on the `Albumentations3D GitHub Discussions <https://github.com/jjmcintosh/albumentations3d/discussions>`_ page.

Congratulations! You have successfully installed Albumentations3D. You can now proceed to the :doc:`Getting Started <getting.started>` guide to learn how to use Albumentations3D for 3D image augmentation.