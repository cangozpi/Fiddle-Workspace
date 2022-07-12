package com.example.demo.security.services.MyAuthenticationProviders;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.authentication.AuthenticationProvider;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;

@Component
public class JwtAuthenticationProvider implements AuthenticationProvider {

    // Injecting available encryption bean
    @Autowired
    private PasswordEncoder passwordEncoder;

    // Injecting our custom UserDetailsService implementation
    @Autowired
    private UserDetailsService userDetailsService;

    @Override
    public Authentication authenticate(Authentication authentication) throws AuthenticationException {
        // In BasicController.login() method, we call authenticationManager.authenticate(token)
        // Then, Authentication Manager calls AuthenticationProvider's authenticate method.
        // Since JwtAuthenticationProvider is our custom authentication provider,
        // this method will be executed.
        String username = authentication.getName();
        String password = String.valueOf(authentication.getCredentials());

        // Fetching user as wrapped with UserDetails object
        UserDetails userDetails = userDetailsService.loadUserByUsername(username);

        // If user is not null, then we check if password matches
        if (userDetails != null){
            if (passwordEncoder.matches(password, userDetails.getPassword())){
                // if it matches, then we can initialize UsernamePasswordAuthenticationToken.
                // Attention! We used its 3 parameters constructor.
                UsernamePasswordAuthenticationToken authenticationToken =
                        new UsernamePasswordAuthenticationToken(username, password, userDetails.getAuthorities());
                return authenticationToken;
            }
        }
        throw new BadCredentialsException("Error!!");
    }

    // Authentication Manager checks if the token is supported by this filter
    // to avoid unnecessary checks.
    @Override
    public boolean supports(Class<?> authenticationType) {
        return UsernamePasswordAuthenticationToken.class.equals(authenticationType);
    }

}
