package com.leousum.boot.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * ClassName HelloController
 * Description
 * Create by 廖双
 * Date 2023/7/25 17:19
 */
@RestController //合成注解RestController等于ResponseBody + Controller
public class HelloController {
    @GetMapping("/hello")
    public String Hello(){
        return "Hello World!";
    }
}
