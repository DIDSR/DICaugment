Installation
=============================

This guide will walk you through the installation process for DICaugment, a Python package for 3D image augmentation specifically designed for volumetric 3D images like CT scans.






Prerequisites
-------------------------

Before installing DICaugment, ensure you have the following prerequisites:

- Python 3.8 or higher

Installation Steps
-------------------------

Follow these steps to install dicaugment:

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

4. Install DICaugment and its dependencies using `pip`:

    .. code-block:: bash

        pip install dicaugment


This will install DICaugment along with the required dependencies.

5. Verify the installation by importing DICaugment in Python:

    .. code-block:: python

        import dicaugment as dca


If no errors occur, the installation was successful.

Upgrade to the Latest Version
-----------------------------------------

To upgrade to the latest version of DICaugment, use the `--upgrade` flag with `pip`:

    .. code-block:: bash
        
        pip install --upgrade dicaugment


This will update DICaugment to the latest available version.

Uninstall DICaugment
----------------------------------------

To uninstall DICaugment, use the following `pip` command:

    .. code-block:: bash
        
        pip uninstall dicaugment

Confirm the uninstallation when prompted.

Additional Notes
------------------------------

- It's recommended to install DICaugment in a virtual environment to isolate it from other Python packages and prevent conflicts.

- If you encounter any issues during installation, please seek help from the dicaugment community on the `DICaugment GitHub Discussions <https://github.com/jjmcintosh/dicaugment/discussions>`_ page.

Congratulations! You have successfully installed DICaugment. You can now proceed to the :doc:`Getting Started <getting.started>` guide to learn how to use DICaugment for 3D image augmentation.