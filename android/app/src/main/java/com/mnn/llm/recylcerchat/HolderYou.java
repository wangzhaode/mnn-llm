package com.mnn.llm.recylcerchat;

import android.view.View;
import android.widget.TextView;

import androidx.recyclerview.widget.RecyclerView;

import com.mnn.llm.R;

public class HolderYou extends RecyclerView.ViewHolder {

    private TextView chatText;

    public HolderYou(View v) {
        super(v);
        chatText = (TextView) v.findViewById(R.id.tv_chat_text);
    }

    public TextView getChatText() {
        return chatText;
    }

    public void setChatText(TextView chatText) {
        this.chatText = chatText;
    }
}
