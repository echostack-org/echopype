"""
Colormaps for plotting echograms.

Importing this package (i.e., ``import echopype.colormaps``) adds echogram-specific colormaps to
Matplotlib's existing colormaps. These
always start with ``ep.`` and come in pairs (the colormap and a reversed version with a name
ending in ``_r``).

Echopype currently provides two colormaps, ``ep.ek500`` and ``ep.ek500_r``, based on the
colormap that the Simrad EK500 echosounder used.

Notes
-----
To use this subpackage the ``matplotlib`` package must be installed.

Examples
--------
The list of colormaps added can be found with this code:

.. code-block:: python

    import echopype.colormap
    from matplotlib import colormaps
    print([name for name in colormaps if name.startswith('ep.')])
    
.. code-block:: text
   :caption: Output

   ['ep.ek500', 'ep.ek500_r']
    
"""

from . import cm

__all__ = ["cm"]
