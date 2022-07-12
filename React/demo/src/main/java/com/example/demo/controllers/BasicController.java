package com.example.demo.controllers;

import com.example.demo.security.services.JWTUtil;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class BasicController {

    @Autowired
    private JWTUtil jwtUtil;

    // injecting authentication manager
    @Autowired
    private AuthenticationManager authenticationManager;

    @PostMapping("/login")
    public ResponseEntity<String> login(@RequestBody LoginRequestDTO request) {
        // Creating UsernamePasswordAuthenticationToken object
        // to send it to authentication manager.
        // Attention! We used two parameters constructor.
        // It sets authentication false by doing this.setAuthenticated(false);
        UsernamePasswordAuthenticationToken token = new UsernamePasswordAuthenticationToken(request.getUsername(), request.getPassword());
        // we let the manager do its job.
        authenticationManager.authenticate(token);
        // if there is no exception thrown from authentication manager,
        // we can generate a JWT token and give it to user.
        String jwt = jwtUtil.generate(request.getUsername());
        return ResponseEntity.ok(jwt);
    }

    @GetMapping("/restrictedAPI")
    public ResponseEntity<String> get(){
        return ResponseEntity.ok("Authorization with JWT token successfull");
    }

}

class LoginRequestDTO {
    private String username;
    private String password;

    public LoginRequestDTO() {
    }

    public String getUsername() {
        return username;
    }

    public String getPassword() {
        return password;
    }
}
