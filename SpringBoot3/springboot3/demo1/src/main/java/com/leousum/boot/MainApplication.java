package com.leousum.boot;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * ClassName MainApplication
 * Description 启动springboot项目的主入口程序
 * Create by 廖双
 * Date 2023/7/25 17:10
 */

@SpringBootApplication //声明这是一个SpringBoot应用
public class MainApplication {
    public static void main(String[] args) {
        SpringApplication.run(MainApplication.class, args);
    }
}
