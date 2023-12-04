if [ $# -lt 1 ]; then
    echo 'Usage: ./package.sh $package_path'
    exit 1
fi

package_path=$1

# make dir
mkdir -p $package_path
cd $package_path
mkdir resource
mkdir build
cd ..

# copy file
echo 'copy files ...'
cp -r script $package_path
cp build/*_demo $package_path/build
cp resource/prompt.txt $package_path/build
# linux
cp libs/*.so build/*.so $package_path/build 2> /dev/null || :
# macos
cp libs/*.dylib build/*.dylib $package_path/build 2> /dev/null || :
