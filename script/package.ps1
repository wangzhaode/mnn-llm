Param(
    [String]$package_path
)

mkdir $package_path
cd $package_path
mkdir resource
mkdir build
cd ..

# copy file
echo 'copy files ...'
cp -r resource\tokenizer $package_path\resource
cp -r script $package_path
cp -r build\Release $package_path\build
cp libs\*.dll $package_path\build\Release
cp libs\*.lib $package_path\build\Release