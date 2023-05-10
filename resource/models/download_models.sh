
help(){
	echo "model download script"
	echo "Usage: ./download_model.sh [OPTIONS]" 

	echo "Options:"
  	echo "    -h,--help                   Print this help message and exit"
  	echo "    fp16,int8,int4              Chose different models"
  	echo "    proxy                       Use https://ghproxy.com to proxy github when download file"
}

fp16_model(){
	for i in `seq 0 27`
   do
      wget -c $1https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.1/glm_block_$i.mnn -P fp16
   done

   wget -c $1https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.1/lm.mnn -P fp16
   wget -c $1https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.1/slim_word_embeddings.bin -P fp16
}

int8_model(){
   for i in `seq 0 27`
   do
      wget -c $1https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.2/glm_block_$i.mnn -P int8
   done

   wget -c $1https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.1/lm.mnn -P int8
   wget -c $1https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.1/slim_word_embeddings.bin -P int8
}

int4_model(){
   for i in `seq 0 27`
   do
      wget -c $1https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.3/glm_block_$i.mnn -P int4
   done

   wget -c $1https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.1/lm.mnn -P int4
   wget -c $1https://github.com/wangzhaode/ChatGLM-MNN/releases/download/v0.1/slim_word_embeddings.bin -P int4
}

if [ $# -eq 0 ]
then
   help
   exit 1
fi

proxy=false

for arg in "$@"
do
    case $arg in
        "--help" )
           help;;
        "-h" )
           help;;
        "proxy" )
           proxy=true;;
   esac
done

if [ "$proxy" = true ]; then
   for arg in "$@"
      do
         case $arg in
               "fp16" )
                  fp16_model "https://ghproxy.com/";;
               "int8" )
                  int8_model "https://ghproxy.com/";;
               "int4" )
                  int4_model "https://ghproxy.com/";;
                * )
                   help;;
         esac
      done
else
   for arg in "$@"
      do
         case $arg in
              "fp16" )
                 fp16_model "";;
              "int8" )
                 int8_model "";;
              "int4" )
                 int4_model "";;
               * )
                  help;;
         esac
      done
fi
