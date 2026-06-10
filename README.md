# Open quantum systems with Tensor Networks - Lake Como TQD 2026 Summer School

##  Getting started with the Notebooks

All tensor network calculations in this summer school use the
using the `Julia` version of the **ITensor** library. You can get
detailed instructions on its installation at:\
<https://itensor.github.io/ITensors.jl/stable/getting_started/Installing.html>\
These notes give you a summary of the steps and should be sufficient,
but consult the above link if you encounter issues.

## Required software

In this lab class you will require the following software to be
installed on your laptop:

-   `Jupyter` notebooks\
    If you are a `Python` user then you will almost certainly
    have this installed already. Otherwise, I would recommend installing
    `Anaconda` which then sets up an entire suite of software tools for
    `Python` and other languages including `Jupyter` and `VScode`.

-   `Julia`\
    This is the programming language we will be using in this class. No
    prior knowledge or experience is required. It is a very similar
    language to `Python` and `Matlab` but can be compiled and runs fast.
    To install `Julia` visit:\
    <https://julialang.org/downloads/>.

## Packages required

Once you have installed `Jupyter` and `Julia` we will need to add some
packages to `Julia`. You may need to run the following lines in the
terminal (for a Mac) to create a soft link to the new installation

``` {.cmd language="cmd"}
your_mac$ rm -f /usr/local/bin/julia
your_mac$ ln -s /Applications/Julia-1.X.X.app/Contents/Resources/julia/bin/julia /usr/local/bin/julia
```

replacing "1.X.X\" by whatever latest version number you installed. This
will then allow you to start `Julia` inside a terminal and launch an
interactive `Julia` session (also known as REPL) in a terminal. The
following will appear:

``` {.cmd language="cmd"}
_
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.X.X (20XX-XX-XX)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia> 
```

On the command prompt press the `]` button to switch to the package
manager as:

``` {.cmd language="cmd"}
julia> ]
(@v1.X) pkg> 
```

We can now use the `add` command to install the additional packages we
need. Let's start with the package which allows `Julia` code to be
executed inside a `Jupyter` notebook called `IJulia`. Start by typing
the first line here:

``` {.cmd language="cmd"}
(@v1.X) pkg> add IJulia
Updating registry at `~/.julia/registries/General.toml`
Resolving package versions...
Installed Parsers ------------- v2.5.8
  ...
Precompiling project...
  10 dependencies successfully precompiled in 16 seconds (36 already precompiled)
```

You should see lots of output appear showing downloading of `IJulia` and
other dependencies leading to a successful installation. We can now
repeat this for all the other packages we need, starting with ITensor
as:

``` {.cmd language="cmd"}
(@v1.X) pkg> add ITensors
```

Note the package is called ITensors not ITensor. Next, we install an
add-on library to ITensor to include matrix product state (MPS)
algorithms:

``` {.cmd language="cmd"}
(@v1.X) pkg> add ITensorMPS
...
(@v1.X) pkg> add Observers
...
(@v1.X) pkg> add PackageCompiler
```

We also need to install packages for doing linear algebra and plotting
as:

``` {.cmd language="cmd"}
(@v1.X) pkg> add LinearAlgebra
...
(@v1.X) pkg> add Plots
...
```

## Ready to start

Start up a `Jupyter` notebook session. We can check all the above has
worked by pressing the "New\" button. A tab appears and under
"Notebook:\" it should include `Julia 1.X.X`kernel, as well as the usual `Python 3 (ipykernel)` option. This
confirms the `IJulia` integration with `Jupyter` has worked. Download
and copy the notebooks into a directory on your laptop and
navigate to it. You can now open the first notebook and begin exploring them!
