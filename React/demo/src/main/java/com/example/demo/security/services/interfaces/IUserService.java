package com.example.demo.security.services.interfaces;

import com.example.demo.security.models.User;
import com.example.demo.security.models.UserDto;

import java.util.List;

public interface IUserService {

    User save(UserDto user);
    List<User> findAll();
    User findOne(String username);

}
