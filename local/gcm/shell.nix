{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.gcc
    pkgs.python312
    pkgs.python312Packages.pip
    pkgs.python312Packages.setuptools
    pkgs.python312Packages.wheel
    pkgs.python312Packages.transformers
    pkgs.python312Packages.pytorch
    pkgs.python312Packages.accelerate
    pkgs.python312Packages.hf-xet
    pkgs.python312Packages.huggingface-hub
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.gcc.cc.lib}/lib:$LD_LIBRARY_PATH
  '';
}
