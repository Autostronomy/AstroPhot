============================
Configuration File Interface
============================

Basic usage
-----------

While the scripting interface is the most versatile, a configuration file interface is included for more striaghtforward workflows. Once a configuration file is created, it can be run simply with::

  ~$ astrophot configfile.py

In this way it can be very easy to pass configuration file between researchers.

Optional flags
--------------

Here we list the optional flags that can be passed at the command line to alter the runtime behaviour of AstroPhot.

Running ``astrophot --help`` will generate the following message::

  usage: astrophot [-h] [--config format] [-v] [--log logfile.log] [-q]
                [--dtype datatype] [--device device]
                [configfile]

  Fast and flexible astronomical image photometry package. For the documentation go to: https://github.com/Autostronomy/AstroPhot

  positional arguments:
    configfile         the path to the configuration file. Or just 'tutorial' to download tutorials.

  optional arguments:
    -h, --help         show this help message and exit
    --config format    The type of configuration file being being provided. One
                       of: astrophot, galfit.
    -v, --version      print the current AstroPhot version to screen
    --log logfile.log  set the log file name for AstroPhot. use 'none' to
                       suppress the log file.
    -q                 quiet flag to stop command line output, only print to log
                       file
    --dtype datatype   set the float point precision. Must be one of: float64,
                       float32
    --device device    set the device for AstroPhot to use for computations. Must
                       be one of: cpu, gpu

  Please see the documentation or contact connor stone
  (connorstone628@gmail.com) for further assistance.


Example config files
--------------------

Further examples are under construction. A basic config file will be downloaded along with the other tutorials listed in the :doc:`getting_started` module. You can run the config file as described above to see it in action.
