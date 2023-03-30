
help(){
	echo "model download script"
	echo "Usage: ./download_model.sh [OPTIONS]" 

	echo "Options:"
  	echo "	-h,--help               Print this help message and exit"
  	echo "fp16,int8,int4	            Chose different models"
}

fp16_model(){
	mkdir -p fp16
	for i in `seq 0 27`
   do
      wget -c https://ghproxy.com/https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.1/glm_block_$i.mnn
   done

   wget -c https://ghproxy.com/https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.1/lm.mnn
   wget -c https://ghproxy.com/https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.1/slim_word_embeddings.bin
}

int8_model(){
   mkdir -p int8
   for i in `seq 0 27`
   do
      wget -c https://ghproxy.com/https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.2/glm_block_$i.mnn
   done

   wget -c https://ghproxy.com/https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.1/lm.mnn
   wget -c https://ghproxy.com/https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.1/slim_word_embeddings.bin
}

int4_model(){
   mkdir -p int4
   for i in `seq 0 27`
   do
      wget -c https://ghproxy.com/https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.3/glm_block_$i.mnn
   done

   wget -c https://ghproxy.com/https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.1/lm.mnn
   wget -c https://ghproxy.com/https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.1/slim_word_embeddings.bin
}

if [ $# -eq 0 ]
then
   help
   exit 1
fi

for arg in "$@"
do
    case $arg in
        "--help" )
           help;;
        "-h" )
           help;;
        "fp16" )
           fp16_model;;
        "int8" )
           int8_model;;
        "int4" )
           int4_model;;
         * )
            help;;
   esac
done
