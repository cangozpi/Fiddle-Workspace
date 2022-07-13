package com.example.demo.security.services.interfaces;

import com.example.demo.security.models.Role;

public interface IRoleService {

    Role findByName(String name);

}
