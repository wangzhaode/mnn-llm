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
cp -r script $package_path
cp resource\prompt.txt $package_path\build
cp -r build\Release $package_path\build
cp libs\*.dll $package_path\build\Release
cp libs\*.lib $package_path\build\Release
