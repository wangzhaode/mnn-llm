package com.mnn.llm;

import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.mnn.llm.recylcerchat.ChatData;
import com.mnn.llm.recylcerchat.ConversationRecyclerView;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class Conversation extends BaseActivity {

    private RecyclerView mRecyclerView;
    private ConversationRecyclerView mAdapter;
    private EditText text;
    private Button send;
    private DateFormat mDateFormat;
    private Chat mChat;
    private boolean mHistory = true;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_conversation);
        mChat = (Chat) getIntent().getSerializableExtra("chat");
        mDateFormat = new SimpleDateFormat("hh:mm aa");
        setupToolbarWithUpNav(R.id.toolbar, "mnn-llm", R.drawable.ic_action_back);

        mRecyclerView = (RecyclerView) findViewById(R.id.recyclerView);
        mRecyclerView.setHasFixedSize(true);
        mRecyclerView.setLayoutManager(new LinearLayoutManager(this));
        mAdapter = new ConversationRecyclerView(this, initData());
        mRecyclerView.setAdapter(mAdapter);
        mRecyclerView.postDelayed(new Runnable() {
            @Override
            public void run() {
                mRecyclerView.smoothScrollToPosition(mRecyclerView.getAdapter().getItemCount() - 1);
            }
        }, 1000);

        text = (EditText) findViewById(R.id.et_message);
        text.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                mRecyclerView.postDelayed(new Runnable() {
                    @Override
                    public void run() {
                        mRecyclerView.smoothScrollToPosition(mRecyclerView.getAdapter().getItemCount() - 1);
                    }
                }, 500);
            }
        });
        send = (Button) findViewById(R.id.bt_send);
        send.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String inputString = text.getText().toString();
                if (!inputString.equals("")){
                    ChatData item = new ChatData();
                    item.setTime(mDateFormat.format(new Date()));
                    item.setType("2");
                    item.setText(inputString);
                    mAdapter.addItem(item);
                    mRecyclerView.smoothScrollToPosition(mRecyclerView.getAdapter().getItemCount() -1);
                    text.setText("");

                    if (inputString.equals("/reset")) {
                        mChat.Reset();
                    } else {
                        // response
                        ChatData response = new ChatData();
                        response.setTime(mDateFormat.format(new Date()));
                        response.setType("1");
                        response.setText("");
                        mAdapter.addItem(response);
                        mRecyclerView.smoothScrollToPosition(mRecyclerView.getAdapter().getItemCount() -1);
                        Handler responseHandler = new Handler() {
                            @Override
                            public void handleMessage(Message msg) {
                                super.handleMessage(msg);
                                ChatData response = new ChatData();
                                response.setTime(mDateFormat.format(new Date()));
                                response.setType("1");
                                response.setText(msg.obj.toString());
                                mAdapter.updateRecentItem(response);
                            }
                        };
                        ResponseThread responseT = new ResponseThread(mChat, inputString, responseHandler, mHistory);
                        responseT.start();
                    }
                }
            }
        });
    }

    public List<ChatData> initData(){
        List<ChatData> data = new ArrayList<>();
        // set head time: year-month-day
        ChatData head = new ChatData();
        DateFormat headFormat = new SimpleDateFormat("yyyy-MM-dd");
        String headDate = headFormat.format(new Date());
        head.setTime("");
        head.setText(headDate);
        head.setType("0");
        data.add(head);
        // set first item
        ChatData item = new ChatData();
        String itemDate = mDateFormat.format(new Date());
        item.setType("1");
        item.setTime(itemDate);
        item.setText("你好，我是mnn-llm，欢迎向我提问。");
        data.add(item);

        return data;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.menu_userphoto, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        /*
        if (mHistory) {
            Toast.makeText(getBaseContext(), "关闭上下文", Toast.LENGTH_SHORT).show();
        } else {
            Toast.makeText(getBaseContext(), "打开上下文", Toast.LENGTH_SHORT).show();
        }
        mHistory = !mHistory;
        */
        Toast.makeText(getBaseContext(), "清空记忆", Toast.LENGTH_SHORT).show();
        mChat.Reset();
        return true;
    }
}

class ResponseThread extends Thread {
    private String mInput;
    private Handler mHandler;
    private Chat mChat;
    private boolean mHistory;

    ResponseThread(Chat chat, String input, Handler handler, boolean history) {
        mChat = chat;
        mInput = input;
        mHandler = handler;
        mHistory = history;
    }

    public void run() {
        super.run();
        mChat.Submit(mInput);
        String last_response = "";
        System.out.println("[MNN_DEBUG] start response\n");
        while (!last_response.contains("<eop>")) {
            try {
                Thread.sleep(50);
            } catch (Exception e) {}
            String response = new String(mChat.Response());
            if (response.equals(last_response)) {
                continue;
            } else {
                last_response = response;
            }
            Message msg = new Message();
            System.out.println("[MNN_DEBUG] " + response);
            msg.obj = response.replaceFirst("<eop>", "");
            mHandler.sendMessage(msg);
        }
        System.out.println("[MNN_DEBUG] response end\n");
        mChat.Done();
        if (!mHistory) {
            mChat.Reset();
        }
    }
}