# nm_lib

This is intended for student to develop the library of a numerical code following the [excersices](https://github.com/AST-Course/AST5110/).

Please create [AST 5110 wiki](https://github.com/AST-Course/AST5110/wiki) to add a detailed description about this code and what it can do!

### Fork this repository privately:
[Setup private fork](https://github.com/AST-Course/AST5110/wiki/Setup-private-fork)
[to create a fork follow the instructions](https://gist.github.com/0xjac/85097472043b697ab57ba1b1c7530274)

### In case you want to do the course in a different environment, do the following [See setup-python-environment](https://github.com/AST-Course/AST5110/wiki/Setup-python-environment)
:
```
mamba create --name ast5110_course python=3.12
mamba activate ast5110_course
mamba install --file requirements.txt
```

### To install the files:
```
cd nm_lib
pip install -e .
```

### To start using the library:
Run, for instance, this code to get started:
```
from nm_lib.nm_ex import nm_lib_ex_1 as nm1
```
